function fig = apply_scienceplots_style(nrows, ncols, opts)
    %APPLY_SCIENCEPLOTS_STYLE Apply SciencePlots-like defaults in MATLAB.
    %   fig = APPLY_SCIENCEPLOTS_STYLE(style, palette, span)
    %   style:  "science" | "ieee" | "nature" | "notebook"
    %   palette: "std" | "muted" | "vibrant" | "bright" | "high-contrast" |
    %            "high-vis" | "retro" | "light"
    %   span:   "single" | "double" (used for ieee/nature)

    arguments
        nrows = 1
        ncols = 1
        opts.figsize = [3 3]
        opts.style = "science"
        opts.palette = "std"
        opts.span = "single"
        opts.polar = false
        opts.font_size = 16
        opts.line_width = 1.0
        opts.axis_line_width = 0.25
        opts.font_name = "Arial"
        opts.legend_box = "on"
    end

    % Base defaults (science.mplstyle)
    set(groot, ...
        "defaultAxesTickDir","in", ...
        "defaultAxesTickLength",[0.015 0.015], ...
        "defaultAxesXMinorTick","on", ...
        "defaultAxesYMinorTick","on", ...
        "defaultAxesLineWidth",opts.axis_line_width, ...
        "defaultLineLineWidth",opts.line_width, ...
        "defaultLegendBox",opts.legend_box, ...
        "defaultAxesFontName",opts.font_name, ...
        "defaultTextFontName",opts.font_name, ...
        "defaultTextInterpreter","latex", ...
        "defaultAxesFontSize",opts.font_size, ...
        "defaultTextFontSize",opts.font_size);

    if opts.polar
        set(groot, "defaultAxesBox", "off");
    end

    set(groot, "defaultAxesColorOrder", palette_colors(opts.palette));

    switch lower(opts.style)
        case "science"
            set(groot, "defaultAxesXGrid", "off", "defaultAxesYGrid", "off");
            fig = figure("Units", "inches", "Position", [1 1 opts.figsize(1) * ncols opts.figsize(2) * nrows]);

        case "ieee"
            set(groot, "defaultAxesFontSize",8, "defaultTextFontSize",8);
            if opts.span == "double"
                w = 7.16;
            else
                w = 3.3;
            end
            fig = figure("Units","inches","Position",[1 1 w* ncols 2.5* nrows]);

        case "nature"
            set(groot, ...
                "defaultAxesFontName",opts.font_name, ...
                "defaultTextFontName",opts.font_name, ...
                "defaultAxesFontSize",opts.font_size, ...
                "defaultTextFontSize",opts.font_size, ...
                "defaultAxesLineWidth",0.5, ...
                "defaultLineLineWidth",opts.line_width);
            if opts.span == "double"
                w = 183/25.4;
            else
                w = 89/25.4;
            end
            fig = figure("Units","inches","Position",[1 1 w* ncols 2.5* nrows]);

        case "notebook"
            set(groot, ...
                "defaultAxesFontName",opts.font_name, ...
                "defaultTextFontName",opts.font_name, ...
                "defaultAxesFontSize",opts.font_size, ...
                "defaultTextFontSize",opts.font_size, ...
                "defaultAxesLineWidth",1.0, ...
                "defaultLineLineWidth",2.0, ...
                "defaultTextInterpreter","tex");
            fig = figure("Units","inches","Position",[1 1 8* ncols 6* nrows]);

        otherwise
            error("Unknown style: %s", opts.style);
    end
end

function c = palette_colors(p)
    switch lower(p)
        case "std"
            c = hex2rgb(["0C5DA5","00B945","FF9500","FF2C00","845B97","474747","9E9E9E"]);
        case "muted"
            c = hex2rgb(["CC6677","332288","DDCC77","117733","88CCEE","882255","44AA99","999933","AA4499","DDDDDD"]);
        case "vibrant"
            c = hex2rgb(["EE7733","0077BB","33BBEE","EE3377","CC3311","009988","BBBBBB"]);
        case "bright"
            c = hex2rgb(["4477AA","EE6677","228833","CCBB44","66CCEE","AA3377","BBBBBB"]);
        case "high-contrast"
            c = hex2rgb(["004488","DDAA33","BB5566"]);
        case "high-vis"
            c = hex2rgb(["0D49FB","E6091C","26EB47","8936DF","FEC32D","25D7FD"]);
        case "retro"
            c = hex2rgb(["4165C0","E770A2","5AC3BE","696969","F79A1E","BA7DCD"]);
        case "light"
            c = hex2rgb(["77AADD","EE8866","EEDD88","FFAABB","99DDFF","44BB99","BBCC33","AAAA00","DDDDDD"]);
        otherwise
            error("Unknown palette: %s", p);
    end
end

function rgb = hex2rgb(hex)
    % hex  : char vector, string scalar, or string/char array (N×1)
    % rgb  : N×3 double in [0 1]

    if ischar(hex), hex = string(hex); end
    hex = strrep(hex, "#", "");
    N = numel(hex);
    rgb = zeros(N,3);

    for k = 1:N
        h = char(hex(k));
        if numel(h) == 3          % short form e.g. "F80" -> "FF8800"
            h = [h(1) h(1) h(2) h(2) h(3) h(3)];
        end
        if numel(h) ~= 6
            error("hex2rgb:badInput", "Hex color must be 3 or 6 hex digits.");
        end
        bytes = sscanf(h, '%2x');     % returns 3x1 integer
        rgb(k,:) = double(bytes(:)') / 255;
    end
end