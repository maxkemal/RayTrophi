#include "Core/StudioLaunchOptions.h"

#include "globals.h"

#include <charconv>
#include <filesystem>
#include <string_view>

namespace {
bool parseIntArgument(std::string_view text, int minimum, int& out) {
    if (text.empty()) {
        return false;
    }
    int value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc() || parsed.ptr != end || value < minimum) {
        return false;
    }
    out = value;
    return true;
}
} // namespace

namespace StudioShell {

LaunchOptions parseLaunchOptions(int argc, char* argv[]) {
    LaunchOptions options;
    auto& project_to_load = options.project_to_load;
    auto& script_to_run = options.script_to_run;
    auto& cli_render_output = options.cli_render_output;
    auto& cli_render_spp = options.cli_render_spp;
    auto& cli_render_frame = options.cli_render_frame;
    auto& cli_seq_start_frame = options.cli_seq_start_frame;
    auto& cli_seq_end_frame = options.cli_seq_end_frame;
    auto& cli_seq_output_dir = options.cli_seq_output_dir;
    auto& headless_requested = options.headless_requested;
    auto& cli_argument_error = options.cli_argument_error;
    auto& cli_argument_errors = options.cli_argument_errors;
    auto& cli_render_mode = options.cli_render_mode;
    auto& cli_seq_mode = options.cli_seq_mode;
    auto& run_hidden = options.run_hidden;

    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) {
            continue;
        }
        const std::string_view arg(argv[i]);
        if (arg == "--script") {
            if (i + 1 < argc && argv[i + 1]) {
                script_to_run = std::filesystem::absolute(argv[++i]).string();
            } else {
                SCENE_LOG_ERROR("--script requires a Python file path");
                cli_argument_errors.push_back("--script requires a Python file path");
                cli_argument_error = true;
            }
        } else if (arg == "--project") {
            if (i + 1 < argc && argv[i + 1]) {
                project_to_load = std::filesystem::absolute(argv[++i]).string();
            } else {
                SCENE_LOG_ERROR("--project requires an .rtp file path");
                cli_argument_errors.push_back("--project requires an .rtp file path");
                cli_argument_error = true;
            }
        } else if (arg == "--render") {
            if (i + 1 < argc && argv[i + 1]) {
                cli_render_output = std::filesystem::absolute(argv[++i]).string();
            } else {
                SCENE_LOG_ERROR("--render requires an output .png path");
                cli_argument_errors.push_back("--render requires an output .png path");
                cli_argument_error = true;
            }
        } else if (arg == "--spp") {
            if (i + 1 >= argc || !argv[i + 1] ||
                !parseIntArgument(argv[++i], 1, cli_render_spp)) {
                SCENE_LOG_ERROR("--spp requires a positive integer");
                cli_argument_errors.push_back("--spp requires a positive integer");
                cli_argument_error = true;
            }
        } else if (arg == "--frame") {
            if (i + 1 >= argc || !argv[i + 1] ||
                !parseIntArgument(argv[++i], 0, cli_render_frame)) {
                SCENE_LOG_ERROR("--frame requires a non-negative integer");
                cli_argument_errors.push_back("--frame requires a non-negative integer");
                cli_argument_error = true;
            }
        } else if (arg == "--frames") {
            // Accept "N-M" format, e.g. --frames 0-100
            if (i + 1 < argc && argv[i + 1]) {
                const std::string_view range(argv[++i]);
                const auto dash = range.find('-');
                int sf = -1, ef = -1;
                bool parse_ok = dash != std::string_view::npos &&
                    parseIntArgument(range.substr(0, dash), 0, sf) &&
                    parseIntArgument(range.substr(dash + 1), sf, ef);
                if (parse_ok) {
                    cli_seq_start_frame = sf;
                    cli_seq_end_frame   = ef;
                } else {
                    SCENE_LOG_ERROR("--frames requires a range in the form N-M (e.g. 0-100)");
                    cli_argument_errors.push_back("--frames requires a range in the form N-M (e.g. 0-100)");
                    cli_argument_error = true;
                }
            } else {
                SCENE_LOG_ERROR("--frames requires a range argument (e.g. 0-100)");
                cli_argument_errors.push_back("--frames requires a range argument (e.g. 0-100)");
                cli_argument_error = true;
            }
        } else if (arg == "--frames-dir") {
            if (i + 1 < argc && argv[i + 1]) {
                cli_seq_output_dir = std::filesystem::absolute(argv[++i]).string();
            } else {
                SCENE_LOG_ERROR("--frames-dir requires a directory path");
                cli_argument_errors.push_back("--frames-dir requires a directory path");
                cli_argument_error = true;
            }
        } else if (arg == "--headless") {
            headless_requested = true;
        } else if (!arg.empty() && arg.front() != '-' && project_to_load.empty()) {
            project_to_load = std::filesystem::absolute(argv[i]).string();
        }
    }
    cli_render_mode = !cli_render_output.empty();
    cli_seq_mode = cli_seq_start_frame >= 0 && !cli_render_mode;
    run_hidden = cli_render_mode || cli_seq_mode || headless_requested;
    if (headless_requested && !cli_render_mode && !cli_seq_mode) {
        SCENE_LOG_ERROR("--headless requires --render <output.png> or --frames N-M");
        cli_argument_errors.push_back("--headless requires --render <output.png> or --frames N-M");
        cli_argument_error = true;
    }
    if (cli_render_mode && cli_seq_mode) {
        SCENE_LOG_ERROR("--render and --frames cannot be combined; use one or the other");
        cli_argument_errors.push_back("--render and --frames cannot be combined");
        cli_argument_error = true;
    }
    // Default sequence output directory: EXE folder / frames
    if (cli_seq_mode && cli_seq_output_dir.empty()) {
        cli_seq_output_dir = (std::filesystem::current_path() / "frames").string();
    }
    if ((cli_render_mode || cli_seq_mode) && !project_to_load.empty()) {
        std::error_code project_path_error;
        if (!std::filesystem::is_regular_file(project_to_load, project_path_error)) {
            SCENE_LOG_ERROR("CLI project file not found: " + project_to_load);
            cli_argument_errors.push_back("CLI project file not found: " + project_to_load);
            cli_argument_error = true;
        }
    }
    return options;
}

} // namespace StudioShell
