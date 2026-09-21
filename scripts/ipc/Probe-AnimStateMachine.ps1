<#
.SYNOPSIS
  Does the animation state machine actually LEAVE its current state, and does
  the character keep moving while it does?
.DESCRIPTION
  The two ways a state machine fails here are both silent:

  1. STATE NEVER STICKS. The node inspector mirrors the selected asset node
     into the live runtime node every frame through onSave/onLoad, and
     currentStateName used to be part of that payload. A transition completed,
     the next frame put the stale state back, and the machine re-fired the same
     transition forever. Permanently transitioning looks exactly like frozen.
  2. STATE HAS NO POSE. The same mirror rebuilt the node's input pins and the
     rebuilt pins came back with id 0, so every link into the state machine was
     orphaned. Each state then evaluates to an EMPTY pose. Nothing logs, nothing
     errors -- the character simply stops.

  So this probe does not ask "is a state machine present". It asks whether the
  current state CHANGES over a run of frames, whether every state has a pose
  behind it, and whether the clip time keeps advancing across the change.
.NOTES
  The user builds and starts the application and loads a scene with an animated
  character whose graph contains a state machine, before running this.
  The character must have "Use Graph" on -- anim.state_machines reports that as
  an error rather than answering for a graph nothing evaluates.
  The script forces states and puts the original one back.
#>
[CmdletBinding()]
param(
    [string]$Character = '',
    [int]$Frames = 90
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force
$failures = 0
function Check([string]$Name, [bool]$Ok, [string]$Detail = '') {
    if ($Ok) { Write-Host "[PASS] $Name" }
    else {
        Write-Host "[FAIL] $Name" -ForegroundColor Red
        if ($Detail) { Write-Host "       $Detail" -ForegroundColor DarkYellow }
        $script:failures++
    }
}

# The graph is evaluated by the application's own frame loop, not by this pipe.
# A read taken in the same breath as the write that should move it measures the
# PREVIOUS frame, so every sample below is separated by round trips that cost at
# least one frame each.
function Wait-Frame {
    [void](Invoke-RtIpc viewport.status @{})
    [void](Invoke-RtIpc viewport.status @{})
}

try {
    Connect-RtIpc -TimeoutMs 3000

    if (-not $Character) {
        $chars = Invoke-RtIpc anim.characters @{}
        if (-not $chars -or $chars.Count -eq 0) {
            Write-Host '[SKIP] No animated characters in this scene.' -ForegroundColor Yellow
            exit 2
        }
        $Character = $chars[0].name
    }
    Write-Host "       character=$Character"

    $machines = Invoke-RtIpc anim.state_machines @{ character = $Character }
    if (-not $machines -or $machines.Count -eq 0) {
        Write-Host '[SKIP] This character has no state machine node in its runtime graph.' -ForegroundColor Yellow
        exit 2
    }
    $sm = $machines[0]
    $startState = $sm.current_state
    Write-Host "       node_id=$($sm.node_id) current=$startState states=$($sm.states.Count) transitions=$($sm.transitions.Count)"

    Check 'State machine has a current state' ([bool]$sm.current_state) 'An empty current state means every state name failed to resolve; the machine returns an empty pose and the character freezes.'

    # THE quiet one. A state with no pose behind it is not an error at any
    # layer: it evaluates to an empty pose, the renderer skips the write, and
    # the character holds its last frame. Nobody reports this as a bug.
    $orphan = @($sm.states | Where-Object { -not $_.pose_connected })
    $orphanNames = ($orphan | ForEach-Object { $_.name }) -join ', '
    Check 'Every state has a pose source' ($orphan.Count -eq 0) "States with no pose: $orphanNames -- these evaluate to an EMPTY pose, which reads as 'the animation stopped'."

    Check 'At least one transition is authored' ($sm.transitions.Count -gt 0) 'With no transitions the machine can only ever play its default state, which is indistinguishable from a broken one.'

    # Sample the live state over a run of frames. Two things are read together:
    # which state is live, and whether the clip time is advancing. A machine
    # stuck mid-transition reports transitioning=true on every sample while the
    # pose never reaches the renderer.
    $seenStates = @{}
    $transitioningSamples = 0
    $times = @()
    for ($i = 0; $i -lt $Frames; $i++) {
        Wait-Frame
        $s = (Invoke-RtIpc anim.state_machines @{ character = $Character })[0]
        $seenStates[$s.current_state] = $true
        if ($s.transitioning) { $transitioningSamples++ }
        $times += [double](Invoke-RtIpc anim.graph_status @{ character = $Character }).normalized_time
    }

    $distinct = @($seenStates.Keys)
    Write-Host "       states visited: $($distinct -join ', ')"
    Write-Host "       transitioning on $transitioningSamples of $Frames samples"

    # Clip time must move. If it does not, the state machine is not driving a
    # playing clip at all and every state assertion above is measuring a corpse.
    $timeMoved = ($times | Select-Object -Unique).Count -gt 1
    Check 'Clip time advances under the state machine' $timeMoved 'normalized_time never changed -- the graph is paused, or the state pose is empty.'

    # Stuck-in-transition: the re-fire loop reports transitioning on nearly
    # every sample AND never lands anywhere new.
    $stuck = ($transitioningSamples -ge ($Frames - 2)) -and ($distinct.Count -le 1)
    Check 'Machine is not stuck mid-transition' (-not $stuck) 'Transitioning on every sample without ever reaching a new state is the re-fire loop: a stale state is being written back over the runtime one each frame.'

    # Drive it by hand: force_state must actually move the machine. This closes
    # the loop -- if forcing works but nothing changes on its own, the
    # conditions are the problem, not the plumbing.
    if ($sm.states.Count -ge 2) {
        $other = @($sm.states | Where-Object { $_.name -ne $startState })[0].name
        [void](Invoke-RtIpc anim.force_state @{ character = $Character; state = $other })
        Wait-Frame
        $forced = (Invoke-RtIpc anim.state_machines @{ character = $Character })[0].current_state
        Check "force_state moves the machine ($startState -> $other)" ($forced -eq $other) "Machine reports '$forced' after being forced to '$other'. Something is rewriting the runtime state every frame."

        [void](Invoke-RtIpc anim.force_state @{ character = $Character; state = $startState })
        Wait-Frame
    }

    # Left as information, not a check: whether the machine switched states on
    # its own depends on the authored conditions, and a single-transition graph
    # legitimately settles. A FAIL here would be the instrument lying.
    if ($distinct.Count -gt 1) {
        Write-Host '[INFO] Machine changed state on its own during the run.' -ForegroundColor Green
    } else {
        Write-Host "[INFO] Machine stayed in '$startState' for the whole run. Check the transition conditions:" -ForegroundColor Yellow
        foreach ($t in $sm.transitions) {
            $exitNote = if ($t.has_exit_time) { "exit_time=$($t.exit_time)" } else { 'no exit time' }
            Write-Host "       $($t.from) -> $($t.to)  cond=$($t.condition) param='$($t.parameter)' val=$($t.compare_value) $exitNote"
        }
        Write-Host '       Drive a condition with anim.set_graph_param / anim.trigger_graph_param and run again.' -ForegroundColor DarkYellow
    }

    if ($sm.recent_events) {
        Write-Host '       recent graph events:'
        foreach ($e in $sm.recent_events) { Write-Host "         $e" }
    }
}
finally {
    Disconnect-RtIpc
}

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host 'All checks passed.' -ForegroundColor Green
exit 0
