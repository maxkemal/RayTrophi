#!/usr/bin/env python3
"""
RayTrophi Studio - local backup.

★★★ WHAT THIS EXISTS TO CATCH

Copying `source/` by hand produces a backup that is both too big and too small.
Too big because `x64/` rides along; too small because the things git cannot see
are exactly the things no remote can restore. This repo has three such gaps,
each created by a blanket rule in .gitignore that was right for its own reason:

  1. scripts/ipc/*.ps1  - Start-RayTrophi.ps1, rt.ps1, Probe-*.ps1. The harness
     CLAUDE.md tells every agent to drive the app with. Excluded by `*.ps1`.
     ★ RtIpc.psm1 IS tracked - .psm1 does not match that rule. Half the harness
     is in git and half is not, which is worse than either whole state: a fresh
     clone gets the module and no way to launch the app.
  2. .agent/ and the root-level *_plan.md / *_todo.md notes - analyses and
     postmortems. CLAUDE.md 3b calls these the most expensive knowledge here.
  3. .claude/, .github/, .vscode/, .agents/ - the agent and editor configuration
     that makes this checkout behave like this checkout.

Everything git DOES track is copied from the WORKING TREE, not from HEAD, so
uncommitted edits and brand-new files land in the archive too. That is the whole
point: a backup holding only what you already pushed protects nothing.

What is deliberately left out, and why:
  x64/ .vs/ obj/ ptx/ *.obj     build output - reproducible
  vcpkg/ external/ libs/        third-party trees - reproducible, gigabytes
  __pycache__/ *.pyc            derived
  *.zip                         previous backups; never nest them
  RayTrophiAgent/.env           SECRETS. --with-secrets includes it, and the
                                manifest then says so in capitals.
  .git/                         428 MB of history that already lives on GitHub.
                                --with-history adds it as a single git bundle.

Usage:
    python scripts/make_backup.py --dry-run       # list what would go in
    python scripts/make_backup.py                 # write the archive
    python scripts/make_backup.py --with-history  # + full git bundle
    python scripts/make_backup.py --out D:/yedek
"""
import argparse
import datetime
import os
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ignored by git, but real work. Directories are taken whole (minus PRUNE_DIRS).
EXTRA_PATHS = [
    "scripts/ipc",
    ".agent",
    ".agents",
    ".github",
    ".vscode",
    ".claude/settings.json",
    ".claude/settings.local.json",
    "docs/dev/mesh_paint_uvless_plan.md",
    "sculpt_control_graph_todo.md",
    "shared_texture_pool_refactor_plan.md",
    "vulkan_blas_batch_fix.md",
    "scripts/update_readme_stats.ps1",
    "RayTrophiStudio/source/src/_Unused",
]

SECRET_PATHS = ["RayTrophiAgent/.env"]

PRUNE_DIRS = {"__pycache__", ".git", ".vs", "x64", "obj", "Debug", "Release", "node_modules"}
PRUNE_SUFFIX = (".pyc", ".obj", ".zip", ".log", ".ptx", ".ilk", ".pdb", ".idb")

# ★ Vendored third-party trees. These ARE tracked, so `git ls-files` hands them
# over, and they are 60% of the repo by weight - ozz-animation/media alone is
# 85 MB of sample glTF that nothing here builds against. --lean drops them.
# The trade is real and stated in the manifest: a lean archive is NOT
# self-contained. It restores from `git clone` + this zip, not from this zip
# alone, so it is a backup of YOUR work, not of a buildable checkout.
VENDORED_PREFIXES = (
    "RayTrophiStudio/external/",
    "RayTrophiStudio/libs/",
)

# The runtime asset drop: ~500 MB of .glb/.vdb the app loads at runtime, placed
# here by hand rather than produced by the build. git does not track them (x64/
# is ignored, and rightly so) and a "Clean Solution" would remove them.
#
# OFF by default, and that is the maintainer's call, not an oversight: these are
# collected assets he can put back, so a nightly that grows from 20 MB to 450 MB
# to protect them buys nothing and would stop being taken. The script reports the
# size it is skipping so the decision stays visible, and does NOT nag about it.
#
# When asked for, they go in a SEPARATE zip - their update rate is nothing like
# the code's, so pairing them in one archive would waste most of every run.
RUNTIME_ASSET_DIR = "x64/Release/assets"


def git(*args):
    out = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("git " + " ".join(args) + " failed: " + out.stderr.strip())
    return out.stdout


def git_paths():
    """Tracked files PLUS untracked-not-ignored ones.

    ★ The second half matters more than the first: a .cpp written today is
    untracked, and a backup that skips it loses the only copy that exists.
    """
    tracked = git("ls-files", "-z").split("\0")
    fresh = git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    return [p for p in dict.fromkeys(tracked + fresh) if p]


def walk_extra(rel):
    absolute = os.path.join(REPO, rel)
    if os.path.isfile(absolute):
        yield rel
        return
    if not os.path.isdir(absolute):
        return
    for root, dirs, files in os.walk(absolute):
        dirs[:] = [d for d in dirs if d not in PRUNE_DIRS]
        for name in files:
            if name.endswith(PRUNE_SUFFIX):
                continue
            full = os.path.join(root, name)
            yield os.path.relpath(full, REPO).replace("\\", "/")


def runtime_asset_files():
    """Every file under the runtime asset dir, with its size."""
    root = os.path.join(REPO, RUNTIME_ASSET_DIR)
    if not os.path.isdir(root):
        return []
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in PRUNE_DIRS]
        for name in files:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, REPO).replace("\\", "/")
            try:
                found.append((rel, os.path.getsize(full)))
            except OSError:
                pass
    return found


def write_runtime_asset_zip(out_dir, stamp, assets):
    """A separate archive, and deliberately barely compressed.

    .glb and .vdb are already-compressed containers; deflate spends minutes to
    win a few percent. Level 1 keeps the run short enough that the backup is
    actually taken, which is the only property that matters here.
    """
    path = os.path.join(out_dir, "RayTrophi_runtime_assets_" + stamp + ".zip")
    total = sum(size for _, size in assets)
    print("calisma zamani varliklari yaziliyor -> %s (%.0f MB, birkac dakika)"
          % (path, total / 1048576.0))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for rel, _ in assets:
            z.write(os.path.join(REPO, rel), rel)
        z.writestr("RUNTIME_ASSETS_README.txt",
                   "These files were archived from " + RUNTIME_ASSET_DIR + ".\n"
                   "git has never tracked them: x64/ is build output and is ignored.\n"
                   "A 'Clean Solution' deletes this directory. Restore by unzipping\n"
                   "this archive at the repo root - the paths inside are already\n"
                   "relative to it.\n\n"
                   "LICENCE: the vegetation assets may be shared but NEVER sold.\n"
                   "See RayTrophiStudio/assets/THIRD_PARTY_ASSETS.md.\n")
    return path


def manifest_text(paths, extras, secrets_included, history_included, missing, lean,
                  runtime_assets, runtime_zip):
    head = git("rev-parse", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    dirty = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
    history_note = ("INCLUDED (repo.bundle)" if history_included
                    else "NOT included - it is on GitHub origin/main")
    lines = [
        "RayTrophi Studio - backup manifest",
        "created      : " + datetime.datetime.now().isoformat(timespec="seconds"),
        "repo         : " + REPO,
        "branch       : " + branch,
        "HEAD         : " + head,
        "git history  : " + history_note,
        "",
        "files from git (tracked + new untracked): " + str(len(paths)),
        "files git does not see (listed below)   : " + str(len(extras)),
        "",
    ]
    if lean:
        lines += [
            "MODE         : LEAN - vendored third-party trees were NOT archived:",
            "               " + "  ".join(VENDORED_PREFIXES),
            "               This archive is therefore NOT a buildable checkout on its own.",
            "               Restore with:  git clone <origin> && git checkout " + head[:12],
            "               then unzip this over it.",
            "",
        ]
    else:
        lines += ["MODE         : FULL - vendored third-party trees included; unzips to a buildable tree", ""]
    if runtime_assets:
        mb = sum(s for _, s in runtime_assets) / 1048576.0
        if runtime_zip:
            lines += ["runtime assets: %d files, %.0f MB -> %s"
                      % (len(runtime_assets), mb, os.path.basename(runtime_zip)), ""]
        else:
            lines += ["runtime assets: skipped by default - %s holds %d files / %.0f MB "
                      "(--with-runtime-assets)" % (RUNTIME_ASSET_DIR, len(runtime_assets), mb), ""]
    if secrets_included:
        lines += ["*** SECRETS INCLUDED: RayTrophiAgent/.env IS IN THIS ARCHIVE. ***",
                  "*** Do not share or upload this file. ***", ""]
    else:
        lines += ["secrets      : RayTrophiAgent/.env EXCLUDED (--with-secrets to include)", ""]
    if dirty:
        lines += ["UNCOMMITTED at backup time (" + str(len(dirty))
                  + ") - captured from the working tree:"]
        lines += ["  " + d for d in dirty]
        lines += [""]
    else:
        lines += ["working tree was clean at backup time", ""]
    if missing:
        lines += ["listed but MISSING on disk (not archived):"]
        lines += ["  " + m for m in missing]
        lines += [""]
    lines += ["--- files git does not track, archived anyway ---"]
    lines += ["  " + e for e in sorted(extras)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(REPO),
                    help="directory for the archive (default: the repo parent, so it is never inside itself)")
    ap.add_argument("--with-history", action="store_true", help="also archive a full git bundle (--all)")
    ap.add_argument("--with-secrets", action="store_true", help="also archive RayTrophiAgent/.env")
    ap.add_argument("--lean", action="store_true",
                    help="skip vendored third-party (external/, libs/) - ~100 MB smaller, NOT self-contained")
    ap.add_argument("--with-runtime-assets", action="store_true",
                    help="also write a SEPARATE zip of x64/Release/assets (the .glb/.vdb the app loads; ~500 MB)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    paths = git_paths()
    skipped_vendored = 0
    if args.lean:
        kept = [p for p in paths if not p.startswith(VENDORED_PREFIXES)]
        skipped_vendored = len(paths) - len(kept)
        paths = kept
    extras = []
    for rel in EXTRA_PATHS:
        extras.extend(walk_extra(rel))
    if args.with_secrets:
        extras.extend(p for p in SECRET_PATHS if os.path.isfile(os.path.join(REPO, p)))

    everything, missing, total = [], [], 0
    for rel in dict.fromkeys(paths + extras):
        full = os.path.join(REPO, rel)
        if not os.path.isfile(full):
            missing.append(rel)
            continue
        everything.append(rel)
        total += os.path.getsize(full)

    runtime_assets = runtime_asset_files()

    print("git'ten          : %d dosya" % len(paths))
    if args.lean:
        print("lean atlanan     : %d dosya (external/, libs/ - ucuncu parti)" % skipped_vendored)
    print("git'in gormedigi : %d dosya" % len(extras))
    if runtime_assets:
        mb = sum(s for _, s in runtime_assets) / 1048576.0
        tag = "ayri zip'e alinacak" if args.with_runtime_assets else "atlaniyor (--with-runtime-assets)"
        print("calisma varliklari: %d dosya, %.0f MB - %s" % (len(runtime_assets), mb, tag))
    print("arsivlenecek     : %d dosya, %.1f MB (sikistirmadan)" % (len(everything), total / 1048576.0))
    if missing:
        print("diskte yok       : %d (manifest'e yazilacak)" % len(missing))

    if args.dry_run:
        print("\n--- git'in gormedigi, yine de alinacaklar ---")
        for e in sorted(extras):
            print("  " + e)
        print("\n[dry-run] hicbir sey yazilmadi.")
        return 0

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out_zip = os.path.join(args.out, "RayTrophi_backup_" + stamp + ".zip")

    runtime_zip = None
    if args.with_runtime_assets and runtime_assets:
        runtime_zip = write_runtime_asset_zip(args.out, stamp, runtime_assets)

    bundle = None
    if args.with_history:
        bundle = os.path.join(args.out, "RayTrophi_" + stamp + ".bundle")
        print("git bundle olusturuluyor -> " + bundle + " (buyuk, birkac dakika surebilir)")
        git("bundle", "create", bundle, "--all")

    print("yaziliyor -> " + out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in everything:
            z.write(os.path.join(REPO, rel), rel)
        z.writestr("BACKUP_MANIFEST.txt",
                   manifest_text(paths, extras, args.with_secrets, bool(bundle), missing,
                                 args.lean, runtime_assets, runtime_zip))
        if bundle:
            z.write(bundle, "repo.bundle")

    size = os.path.getsize(out_zip)
    print("BITTI: %s  (%.1f MB)" % (out_zip, size / 1048576.0))
    if bundle:
        os.remove(bundle)
        print("       (bundle zip'e girdi, gecici dosya silindi)")
    if runtime_zip:
        print("BITTI: %s  (%.0f MB)" % (runtime_zip, os.path.getsize(runtime_zip) / 1048576.0))
    print("       icindeki BACKUP_MANIFEST.txt neyin neden alindigini/alinmadigini yazar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
