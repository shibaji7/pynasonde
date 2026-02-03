classdef DigiPlots < handle
    %DIGIPLOTS Base plotting helper for Digisonde-style figures.

    properties
        fig_title = ""
        nrows = 1
        ncols = 1
        font_size = 16
        figsize = [3 3]
        date = []
        date_lims = []
        subplot_kw = struct()
        draw_local_time = false
        n_sub_plots = 0
        fig
        tl
        axes
        layout
    end

    methods
        function obj = DigiPlots(fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw, draw_local_time)
            if nargin < 1 || isempty(fig_title), fig_title = ""; end
            if nargin < 2 || isempty(nrows), nrows = 1; end
            if nargin < 3 || isempty(ncols), ncols = 1; end
            if nargin < 4 || isempty(font_size), font_size = 10; end
            if nargin < 5 || isempty(figsize), figsize = [3 3]; end
            if nargin < 6, date = []; end
            if nargin < 7 || isempty(date_lims), date_lims = []; end
            if nargin < 8 || isempty(subplot_kw), subplot_kw = struct(); end
            if nargin < 9 || isempty(draw_local_time), draw_local_time = false; end

            obj.fig_title = fig_title;
            obj.nrows = nrows;
            obj.ncols = ncols;
            obj.font_size = font_size;
            obj.figsize = figsize;
            obj.date = date;
            obj.date_lims = date_lims;
            obj.subplot_kw = subplot_kw;
            obj.draw_local_time = draw_local_time;

            obj.fig = apply_scienceplots_style(nrows, ncols, figsize, "science", "muted");
            obj.tl = tiledlayout(nrows, ncols, 'Padding','loose', 'TileSpacing','loose');
            n_axes = nrows * ncols;
            obj.axes = gobjects(n_axes, 1);
            for i = 1:n_axes
                if isfield(subplot_kw, "projection") && strcmpi(subplot_kw.projection, "polar")
                    ax = polaraxes;
                    ax.ThetaZeroLocation = "top";
                    ax.ThetaDir = "clockwise"; 
                    obj.axes(i) = ax;
                else
                    obj.axes(i) = nexttile;
                end
            end
        end

        function ax = get_axes(obj, del_ticks)
            if nargin < 2 || isempty(del_ticks), del_ticks = true; end
            DigiPlots.setsize(obj.font_size);

            if numel(obj.axes) > 1
                ax = obj.axes(obj.n_sub_plots + 1);
            else
                ax = obj.axes;
            end
            if del_ticks
                if isfield(obj.subplot_kw, "projection") && strcmpi(obj.subplot_kw.projection, "polar")
                    ax.ThetaTick = [];
                    ax.RTick = [];
                else
                    ax.XTick = [];
                    ax.YTick = [];
                end
            end
            if obj.n_sub_plots == 0 && strlength(obj.fig_title) > 0
                text(ax, 0.01, 1.05, obj.fig_title, "Units", "normalized", ...
                    "HorizontalAlignment", "left", "VerticalAlignment", "middle", "FontSize", obj.font_size);
            end
            obj.n_sub_plots = obj.n_sub_plots + 1;
        end

        function save(obj, filepath)
            if endsWith(filepath, ".fig", "IgnoreCase", true);
                saveas(obj.fig, filepath);
            elseif exist("exportgraphics", "file")
                exportgraphics(obj.fig, fullfile(filepath), "ContentType", "vector");
            end
        end

        function close(obj)
            if isgraphics(obj.fig)
                close(obj.fig);
            end
        end

        function add_colorbar(obj, im, ax, label_txt, mpos)
            if nargin < 5 || isempty(mpos), mpos = [0.025 0.0125 0.015 0.5]; end
            if nargin < 4, label_txt = ""; end
        
            % Use normalized units to compute positions
            ax.Units = 'normalized';
            pos = ax.Position;
            cpos = [pos(1) + pos(3) + mpos(1), pos(2) + mpos(2), mpos(3), pos(4) * mpos(4)];
        
            % Create colorbar targeted to axes, set matching Units before Position
            cb = colorbar(ax);
            cb.Units = 'normalized';
            cb.Position = cpos;
        
            % Set label robustly (convert to char if string array/scalar)
            if strlength(string(label_txt)) > 0
                lbl = label_txt;
                if isstring(lbl) || ischar(lbl)
                    % prefer the Label property if present; otherwise fallback to ylabel(cb,...)
                    if isprop(cb,'Label') && ~isempty(cb.Label)
                        cb.Label.String = char(lbl);
                    else
                        ylabel(cb, char(lbl));
                    end
                else
                    % convert other numeric/etc. to string
                    if isscalar(label_txt)
                        txt = char(string(label_txt));
                    else
                        txt = char(string(label_txt));
                    end
                    if isprop(cb,'Label') && ~isempty(cb.Label)
                        cb.Label.String = txt;
                    else
                        ylabel(cb, txt);
                    end
                end
            end
        
            % keep handle to image to prevent optimization away (if needed)
            if nargin >= 2 %#ok<*INUSD>
                % no-op
            end
        end
    end

    methods (Static)
        function setsize(size, ax)
            if nargin < 2 || isempty(ax)
                set(groot, "defaultAxesFontSize", size, "defaultTextFontSize", size);
            else
                ax.FontSize = size;
            end
        end

        function [X, Y, Z] = get_gridded_parameters(tbl, xparam, yparam, zparam, rounding, r)
            if nargin < 5 || isempty(rounding), rounding = true; end
            if nargin < 6 || isempty(r), r = 1; end
            x = tbl.(xparam);
            y = tbl.(yparam);
            z = tbl.(zparam);
            if rounding
                if ~isdatetime(x)
                    x = round(x, r);
                end
                y = round(y, r);
            end
            [G, xg, yg] = findgroups(x, y);
            zmean = splitapply(@mean, z, G);
            ux = unique(xg);
            uy = unique(yg);
            [X, Y] = meshgrid(ux, uy);
            Z = nan(numel(uy), numel(ux));
            [~, ix] = ismember(xg, ux);
            [~, iy] = ismember(yg, uy);
            for k = 1:numel(zmean)
                if ix(k) > 0 && iy(k) > 0
                    Z(iy(k), ix(k)) = zmean(k);
                end
            end
        end

        function cmap = resolve_colormap(cmap_in)
            if isnumeric(cmap_in)
                cmap = cmap_in;
                return
            end
            if isstring(cmap_in) || ischar(cmap_in)
                name = char(cmap_in);
                switch lower(name)
                    case "redblackblue"
                        cmap = DigiPlots.colormap_redblackblue(256);
                    case "inferno"
                        cmap = DigiPlots.colormap_inferno(256);
                    otherwise
                        if exist(name, "file") == 2
                            cmap = feval(name, 256);
                        else
                            cmap = parula(256);
                        end
                end
                return
            end
            cmap = parula(256);
        end

        function colors = sample_colors(base_rgb, n, direction)
            % base_rgb in 0-255 or 0-1
            % direction: "light2dark" or "dark2light"
            if nargin < 2 || isempty(n), n = 6; end
            if nargin < 3 || isempty(direction), direction = "dark2light"; end

            if nargin < 2 || isempty(n), n = 6; end
            base_rgb = DigiPlots.resolve_colormap(base_rgb);
            hsv0 = rgb2hsv(base_rgb/255);
            h = hsv0(1);
            % control saturation and value ramps
            s = linspace(0.4, 0.8, n)';
            v = linspace(0.6, 1.0, n)';

            if strcmpi(direction, "light2dark")
                v = flipud(v);
            end
            colors = hsv2rgb([repmat(h,n,1), s, v]);
        end

        function cmap = colormap_redblackblue(n)
            if nargin < 1 || isempty(n), n = 256; end
            stops = [1 0 0; 0 0 0; 19/255 29/255 227/255];
            cmap = interp1([0 0.5 1], stops, linspace(0, 1, n));
        end

        function cmap = colormap_inferno(n)
            if nargin < 1 || isempty(n), n = 256; end
            stops = [0 0 0; 85/255 0 127/255; 170/255 0 255/255; 1 69/255 0; 1 1 0; 1 1 170/255];
            cmap = interp1([0 0.2 0.4 0.6 0.8 1], stops, linspace(0, 1, n));
        end
    end
end
