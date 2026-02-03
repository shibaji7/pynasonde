function apply_misc_and_language(misc, language)
    %APPLY_MISC_AND_LANGUAGE Apply misc + language equivalents of SciencePlots.
    %   apply_misc_and_language(misc, language)
    %   misc: "no-latex" | "sans" | "latex-sans" | "grid"
    %   language: "russian" | "turkish" | "cjk-sc" | "cjk-tc" | "cjk-jp" | "cjk-kr"

    if nargin < 1, misc = ""; end
    if nargin < 2, language = ""; end

    switch lower(misc)
        case "no-latex"
            set(groot, "defaultTextInterpreter", "tex");
        case "sans"
            set(groot, "defaultAxesFontName", "Arial", ...
                    "defaultTextFontName", "Arial");
        case "latex-sans"
            set(groot, "defaultTextInterpreter", "latex", ...
                    "defaultAxesFontName", "Arial", ...
                    "defaultTextFontName", "Arial");
        case "grid"
            set(groot, ...
                "defaultAxesXGrid", "on", ...
                "defaultAxesYGrid", "on", ...
                "defaultAxesGridLineStyle", "--", ...
                "defaultAxesGridAlpha", 0.5, ...
                "defaultLegendBox", "on");
        case ""
            % no-op
        otherwise
            error("Unknown misc option: %s", misc);
    end

    switch lower(language)
        case "russian"
            set(groot, "defaultAxesFontName", "Times New Roman", ...
                    "defaultTextFontName", "Times New Roman");
        case "turkish"
            set(groot, "defaultAxesFontName", "Arial", ...
                    "defaultTextFontName", "Arial");
        case "cjk-sc"
            set(groot, "defaultAxesFontName", "SimSun", ...
                    "defaultTextFontName", "SimSun");
        case "cjk-tc"
            set(groot, "defaultAxesFontName", "Noto Sans CJK TC", ...
                    "defaultTextFontName", "Noto Sans CJK TC");
        case "cjk-jp"
            set(groot, "defaultAxesFontName", "MS Mincho", ...
                    "defaultTextFontName", "MS Mincho");
        case "cjk-kr"
            set(groot, "defaultAxesFontName", "Malgun Gothic", ...
                    "defaultTextFontName", "Malgun Gothic");
        case ""
            % no-op
        otherwise
            error("Unknown language option: %s", language);
    end
end