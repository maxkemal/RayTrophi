#pragma once

#include <string>
#include <vector>

namespace StudioShell {

// Studio authoring/offline launch policy; not a Player runtime dependency.
struct LaunchOptions {
    std::string project_to_load;
    std::string script_to_run;
    std::string cli_render_output;
    int cli_render_spp = 128;
    int cli_render_frame = -1;
    int cli_seq_start_frame = -1;
    int cli_seq_end_frame = -1;
    std::string cli_seq_output_dir;
    bool headless_requested = false;
    bool cli_argument_error = false;
    std::vector<std::string> cli_argument_errors;
    bool cli_render_mode = false;
    bool cli_seq_mode = false;
    bool run_hidden = false;
};

LaunchOptions parseLaunchOptions(int argc, char* argv[]);

} // namespace StudioShell
