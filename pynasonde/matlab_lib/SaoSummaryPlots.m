classdef SaoSummaryPlots < DigiPlots
    %SAOSUMMARYPLOTS SAO summary time-series and ionogram-style plots.

    methods
        function obj = SaoSummaryPlots(fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw, draw_local_time)
            if nargin < 1, fig_title = ""; end
            if nargin < 2, nrows = 1; end
            if nargin < 3, ncols = 1; end
            if nargin < 4, font_size = 16; end
            if nargin < 5, figsize = [3 3]; end
            if nargin < 6, date = []; end
            if nargin < 7, date_lims = []; end
            if nargin < 8, subplot_kw = struct(); end
            if nargin < 9, draw_local_time = false; end
            obj@DigiPlots(fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw, draw_local_time);
        end

        function [ax, im] = add_TS(obj, df, xparam, yparam, zparam, cbar_label, cmap, prange, ylabel_txt, xlabel_txt, major_locator, minor_locator, ylim, xlim, title_txt, add_cbar, zparam_lim, plot_type, scatter_ms)
            if nargin < 3, xparam = "datetime"; end
            if nargin < 4, yparam = "th"; end
            if nargin < 5, zparam = "pf"; end
            if nargin < 6, cbar_label = "$f_0$, MHz"; end
            if nargin < 7, cmap = "inferno"; end
            if nargin < 8, prange = [1 15]; end
            if nargin < 9, ylabel_txt = "Height, km"; end
            if nargin < 10, xlabel_txt = "Time, UT"; end
            if nargin < 11, major_locator = []; end
            if nargin < 12, minor_locator = []; end
            if nargin < 13, ylim = [80 800]; end
            if nargin < 14, xlim = []; end
            if nargin < 15, title_txt = ""; end
            if nargin < 16 || isempty(add_cbar), add_cbar = true; end
            if nargin < 17, zparam_lim = NaN; end
            if nargin < 18, plot_type = "pcolor"; end
            if nargin < 19, scatter_ms = 4; end
            %#ok<NASGU> % locators not used in MATLAB version

            if obj.draw_local_time
                local_name = "local_" + string(xparam);
                if ismember(local_name, df.Properties.VariableNames)
                    xparam = char(local_name);
                end
                xlabel_txt = strrep(xlabel_txt, "UT", "LT");
            end

            DigiPlots.setsize(obj.font_size);
            ax = obj.get_axes(false);

            xdata = df.(xparam);
            if isempty(xlim)
                xlim = [min(xdata) max(xdata)];
            end
            % ax.XLim = xlim;
            ax.XLabel.String = xlabel_txt;
            ax.YLim = ylim;
            ylabel(ax, ylabel_txt);

            if ~isnan(zparam_lim)
                df = df(df.(zparam) <= zparam_lim, :);
            end

            cmap_arr = DigiPlots.resolve_colormap(cmap);
            colormap(ax, cmap_arr);
            caxis(ax, prange);

            if strcmpi(plot_type, "pcolor")
                [X, Y, Z] = DigiPlots.get_gridded_parameters(df, xparam, yparam, zparam, false, 1);
                if isdatetime(X)
                    Xp = datenum(X);
                    ax.XLim = datenum(xlim);
                else
                    Xp = X;
                end
                im = pcolor(ax, Xp, Y, Z);
                shading(ax, "flat");
                if isdatetime(X)
                    datetick(ax, "x", "HH:MM", "keeplimits", "keepticks");
                end
            else
                im = scatter(ax, df.(xparam), df.(yparam), scatter_ms, df.(zparam), "s", "filled");
            end

            if strlength(title_txt) > 0
                text(ax, 0.95, 1.05, title_txt, "Units", "normalized", "HorizontalAlignment", "right", "VerticalAlignment", "middle");
            end
            if add_cbar
                obj.add_colorbar(im, ax, cbar_label);
            end
        end

        function [ax, tax] = plot_TS(obj, df, xparam, left_yparams, right_yparams, opts)
            arguments
                obj
                df
                xparam
                left_yparams = {}
                right_yparams = {}
                opts.ylabels = {"Frequencies, MHz","Height, km"}
                opts.xlabel_txt = "Time, UT"
                opts.marker = "s"
                opts.major_locator = 60*6
                opts.minor_locator = 60*1
                opts.right_ylim = [100 400]
                opts.left_ylim = [1 15]
                opts.left_yparam_labels = {}
                opts.right_yparam_labels = {}
                opts.xlim = []
                opts.ms = 1
                opts.title_txt = ""
                opts.right_axis_color = []
                opts.left_axis_color = []
                opts.color_map = [255 100 100]
                opts.color_direction = "light2dark"
                opts.txt_pos = [0.9 0.9]
                opts.vlines = []
                opts.vline_style = []
                opts.draw_legend = true
            end

            if obj.draw_local_time
                local_name = "local_" + string(xparam);
                if ismember(local_name, df.Properties.VariableNames)
                    xparam = char(local_name);
                end
                opts.xlabel_txt = strrep(opts.xlabel_txt, "UT", "LT");
            end

            DigiPlots.setsize(obj.font_size);
            ax = obj.get_axes(false);
            
            if isempty(opts.xlim)
                xdata = df.(xparam);
                opts.xlim = [min(xdata) max(xdata)];
            end
            ax.XTick = datenum(opts.xlim(1):minutes(opts.major_locator):opts.xlim(2));
            ax.XMinorTick = "on";
            ax.XAxis.MinorTickValues = datenum(opts.xlim(1):minutes(opts.minor_locator):opts.xlim(2));
            ax.XLim = datenum(opts.xlim);
            xlabel(ax, opts.xlabel_txt);

            colors = DigiPlots.sample_colors(opts.color_map, numel(left_yparams), opts.color_direction);
            if isempty(opts.left_axis_color)
                opts.left_axis_color = colors(1, :);
            end

            if isempty(opts.left_yparam_labels)
                opts.left_yparam_labels = left_yparams;
            end
            if isempty(opts.right_yparam_labels)
                opts.right_yparam_label = right_yparams;
            end

            yyaxis(ax, "left");
            ax.YLim = opts.left_ylim;
            ylabel(ax, opts.ylabels(:,1), "Color", opts.left_axis_color);
            ax.YAxis(1).Color = opts.left_axis_color;
            hold(ax, "on");
            for i = 1:numel(left_yparams)
                yname = left_yparams{i};
                plot(ax, datenum(df.(xparam)), df.(yname), opts.marker, "Color", colors(i, :), ...
                    "MarkerSize", opts.ms, "LineStyle", "none", "DisplayName", opts.left_yparam_labels{i});
                datetick(ax, "x", "HH^{MM}", "keeplimits", "keepticks");
            end
            
            if opts.draw_legend
                legend(ax, "Location", "northwest");
            end

            tax = [];
            if ~isempty(right_yparams)
                yyaxis(ax, "right");
                ax.YLim = opts.right_ylim;
                rcolors = DigiPlots.sample_colors([0, 0, 255], numel(right_yparams), opts.color_direction);
                if isempty(opts.right_axis_color)
                    right_axis_color = rcolors(1, :);
                end
                ylabel(ax, opts.ylabels{2}, "Color", right_axis_color);
                ax.YAxis(2).Color = right_axis_color;
                for i = 1:numel(right_yparams)
                    yname = right_yparams{i};
                    plot(ax, datenum(df.(xparam)), df.(yname), "d", "Color", rcolors(i, :), ...
                        "MarkerSize", opts.ms, "LineStyle", "none", "DisplayName", opts.right_yparam_labels{i});
                end
                tax = ax;
            end
            
            if strlength(opts.title_txt) > 0
                text(ax, opts.txt_pos(1), opts.txt_pos(2), opts.title_txt, "Units", "normalized", "HorizontalAlignment", "right", "VerticalAlignment", "middle");
            end

            if ~isempty(opts.vlines)
                for v = 1:numel(opts.vlines)
                    xv = datenum(opts.vlines(v));
                    yl = ax.YLim;
                    line(ax, [xv xv], yl, "Color", "k", "LineStyle", opts.vline_style, "ZData", [1 1]*ax.ZLim(2));
                end
            end

        end

        function plot_ionogram(obj, df, xparam, yparam, xlabel_txt, ylabel_txt, ylim, xlim, xticks, text_txt, del_ticks, ls, lcolor, lw, zorder, ax, kind)
            if nargin < 3, xparam = "pf"; end
            if nargin < 4, yparam = "th"; end
            if nargin < 5, xlabel_txt = "Frequency, MHz"; end
            if nargin < 6, ylabel_txt = "Virtual Height, km"; end
            if nargin < 7, ylim = [0 600]; end
            if nargin < 8, xlim = [1 22]; end
            if nargin < 9, xticks = [1.5 2.0 3.0 5.0 7.0 10.0 15.0 20.0]; end
            if nargin < 10, text_txt = ""; end
            if nargin < 11, del_ticks = false; end
            if nargin < 12, ls = "-"; end
            if nargin < 13, lcolor = "k"; end
            if nargin < 14, lw = 0.7; end
            if nargin < 15, zorder = 2; end
            if nargin < 16 || isempty(ax), ax = obj.get_axes(del_ticks); end
            if nargin < 17, kind = "ionogram"; end
            %#ok<NASGU>

            DigiPlots.setsize(obj.font_size);
            
            ax.XLim = log10(xlim);
            ax.YLim = ylim;
            if strcmpi(kind, "ionogram")
                plot(ax, log10(df.(xparam)), df.(yparam), "LineStyle", ls, "Color", lcolor, "LineWidth", lw);
            else
                scatter(ax, log10(df.(xparam)), df.(yparam), lw, lcolor, "s", "filled");
            end
            if ~del_ticks
                ax.XTick = log10(xticks);
                ax.XTickLabel = string(xticks);
            end
            if strlength(text_txt) > 0
                text(ax, 0.05, 0.9, text_txt, "Units", "normalized", "HorizontalAlignment", "left", "VerticalAlignment", "middle", "FontSize", obj.font_size);
            end
            xlabel(ax, xlabel_txt, "FontSize", obj.font_size);
            ylabel(ax, ylabel_txt, "FontSize", obj.font_size);

        end
    end

    methods (Static)
        function plot_isodensity_contours(df, xparam, yparam, zparam, xlabel, ylabel, ylim, major_locator, minor_locator, xlim, fbins, text_txt, del_ticks, fname, figsize, lw, alpha, zorder, cmap)
            if nargin < 2, xparam = "date"; end
            if nargin < 3, yparam = "th"; end
            if nargin < 4, zparam = "pf"; end
            if nargin < 5, xlabel = "Time, UT"; end
            if nargin < 6, ylabel = "Virtual Height, km"; end
            if nargin < 7, ylim = [50 300]; end
            if nargin < 8, major_locator = []; end
            if nargin < 9, minor_locator = []; end
            if nargin < 10, xlim = []; end
            if nargin < 11, fbins = [1 2 3 4 5 6 7 8]; end
            if nargin < 12, text_txt = ""; end
            if nargin < 13, del_ticks = false; end
            if nargin < 14, fname = ""; end
            if nargin < 15, figsize = [5 3]; end
            if nargin < 16, lw = 0.7; end
            if nargin < 17, alpha = 0.8; end
            if nargin < 18, zorder = 4; end
            if nargin < 19, cmap = "Spectral"; end
            %#ok<NASGU>

            plotter = SaoSummaryPlots("", 1, 1, 10, figsize);
            ax = plotter.get_axes(del_ticks);
            xdata = df.(xparam);
            if isempty(xlim)
                xlim = [min(xdata) max(xdata)];
            end
            ax.XLim = xlim;
            xlabel(ax, xlabel);
            ax.YLim = ylim;
            ylabel(ax, ylabel);

            for i = 1:(numel(fbins) - 1)
                fmax = fbins(i + 1);
                fmin = fbins(i);
                o = df(df.(zparam) >= fmin & df.(zparam) <= fmax, :);
                scatter(ax, o.(xparam), o.(yparam), 1.5, o.(zparam), "s", "filled", ...
                    "MarkerFaceAlpha", alpha, "MarkerEdgeAlpha", alpha);
                colormap(ax, DigiPlots.resolve_colormap(cmap));
                caxis(ax, [fmin fmax]);
            end

            if strlength(text_txt) > 0
                text(ax, 0.05, 0.9, text_txt, "Units", "normalized", "HorizontalAlignment", "left", "VerticalAlignment", "middle", "FontSize", plotter.font_size);
            end
            if strlength(fname) > 0
                plotter.save(fname);
                plotter.close();
            else
                plotter.close();
            end
        end
    end
end
