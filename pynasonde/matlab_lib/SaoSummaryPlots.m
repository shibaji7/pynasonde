classdef SaoSummaryPlots < DigiPlots
    %SAOSUMMARYPLOTS SAO summary time-series and ionogram-style plots.

    methods
        function obj = SaoSummaryPlots(fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw, draw_local_time)
            arguments
                fig_title = "" 
                nrows = 1 
                ncols = 1
                font_size = 16
                figsize = [3 3]
                date = []
                date_lims = []
                subplot_kw = struct()
                draw_local_time = false
            end
            obj@DigiPlots(fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw, draw_local_time);
        end

        function [ax, im] = add_TS(obj, df, xparam, yparam, zparam, opts)
            arguments
                obj
                df
                xparam = "datetime"
                yparam = "th"
                zparam = "pf"
                opts.cbar_label = "$f_0$, MHz"
                opts.cmap = "inferno"
                opts.prange = [1 15]
                opts.ylabel_txt = "Height, km"
                opts.xlabel_txt = "Time, UT"
                opts.major_locator = 60*6
                opts.minor_locator = 60*1
                opts.ylim = [80 600]
                opts.xlim = []
                opts.title_txt = ""
                opts.add_cbar = false
                opts.zparam_lim = NaN
                opts.plot_type = "pcolor"
                opts.scatter_ms = 4
                opts.ax = []
                opts.txt_pos = [0.9 0.9]
                opts.vlines = []
                opts.vline_styles = []
                opts.date_tick_format = "HH^{MM}"
            end

            if obj.draw_local_time
                local_name = "local_" + string(xparam);
                if ismember(local_name, df.Properties.VariableNames)
                    xparam = char(local_name);
                end
                opts.xlabel_txt = strrep(opts.xlabel_txt, "UT", "LT");
            end

            DigiPlots.setsize(obj.font_size);
            
            if ~isempty(opts.ax)
                ax = opts.ax;
            else
                ax = obj.get_axes(false);
            end
            
            if isempty(opts.xlim)
                xdata = df.(xparam);
                opts.xlim = [min(xdata) max(xdata)];
            end
            ax.XTick = datenum(opts.xlim(1):minutes(opts.major_locator):opts.xlim(2));
            ax.XMinorTick = "on";
            ax.XAxis.MinorTickValues = datenum(opts.xlim(1):minutes(opts.minor_locator):opts.xlim(2));
            ax.XLim = datenum(opts.xlim);
            xlabel(ax, opts.xlabel_txt);
            ax.YLim = opts.ylim;
            ylabel(ax, opts.ylabel_txt);
            if ~isnan(opts.zparam_lim)
                df = df(df.(zparam) <= opts.zparam_lim, :);
            end

            cmap_arr = DigiPlots.resolve_colormap(opts.cmap);
            colormap(ax, cmap_arr);
            caxis(ax, opts.prange);

            if strcmpi(opts.plot_type, "pcolor")
                [X, Y, Z] = DigiPlots.get_gridded_parameters(df, xparam, yparam, zparam, false, 1);
                im = pcolor(ax, datenum(X), Y, Z);
                shading(ax, "flat");
                datetick(ax, "x", opts.date_tick_format, "keeplimits", "keepticks");
                
            else
                im = scatter(ax, df.(xparam), df.(yparam), opts.scatter_ms, df.(zparam), "s", "filled");
            end

            if strlength(opts.title_txt) > 0
                text(ax, opts.txt_pos(1), opts.txt_pos(2), opts.title_txt, "Units", "normalized", "HorizontalAlignment", "right", "VerticalAlignment", "middle");
            end

            if ~isempty(opts.vlines)
                for v = 1:numel(opts.vlines)
                    xv = datenum(opts.vlines(v));
                    yl = ax.YLim;
                    line(ax, [xv xv], yl, "Color", "k", "LineStyle", opts.vline_styles{v}, "ZData", [1 1]*ax.ZLim(2), ...
                            "HandleVisibility", "off");
                end
            end
            if add_cbar
                obj.add_colorbar(im, ax, opts.cbar_label);
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
                opts.markers = ["s" "d"]
                opts.major_locator = 60*6
                opts.minor_locator = 60*1
                opts.right_ylim = [80 400]
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
                opts.vline_styles = []
                opts.draw_legend = true
                opts.ax = []
                opts.date_tick_format = "HH^{MM}"
                opts.dual_frame = false
            end

            if obj.draw_local_time
                local_name = "local_" + string(xparam);
                if ismember(local_name, df.Properties.VariableNames)
                    xparam = char(local_name);
                end
                opts.xlabel_txt = strrep(opts.xlabel_txt, "UT", "LT");
            end

            DigiPlots.setsize(obj.font_size);
            if ~isempty(opts.ax)
                ax = opts.ax;
            else
                ax = obj.get_axes(false);
            end
            
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

            if opts.dual_frame
                yyaxis(ax, "left");
            end
            ax.YLim = opts.left_ylim;
            ylabel(ax, opts.ylabels(:,1), "Color", opts.left_axis_color);
            ax.YAxis(1).Color = opts.left_axis_color;
            hold(ax, "on");
            for i = 1:numel(left_yparams)
                yname = left_yparams{i};
                if opts.draw_legend
                    plot(ax, datenum(df.(xparam)), df.(yname), opts.markers{1}, "Color", colors(i, :), ...
                        "MarkerSize", opts.ms, "DisplayName", opts.left_yparam_labels{i});
                else
                    plot(ax, datenum(df.(xparam)), df.(yname), opts.markers{2}, "Color", colors(i, :), ...
                        "MarkerSize", opts.ms, "HandleVisibility", "off");
                end
                datetick(ax, "x", opts.date_tick_format, "keeplimits", "keepticks");
            end
            hold(ax, "off");
            
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
                hold(ax, "on");
                for i = 1:numel(right_yparams)
                    yname = right_yparams{i};
                    if opts.draw_legend
                        plot(ax, datenum(df.(xparam)), df.(yname), opts.markers{2}, "Color", rcolors(i, :), ...
                            "MarkerSize", opts.ms, "DisplayName", opts.right_yparam_labels{i});
                    else
                        plot(ax, datenum(df.(xparam)), df.(yname), opts.markers{2}, "Color", rcolors(i, :), ...
                            "MarkerSize", opts.ms, "HandleVisibility", "off");
                    end
                end
                hold(ax, "off");
            end
            
            if strlength(opts.title_txt) > 0
                text(ax, opts.txt_pos(1), opts.txt_pos(2), opts.title_txt, "Units", "normalized", "HorizontalAlignment", "right", "VerticalAlignment", "middle");
            end

            if ~isempty(opts.vlines)
                for v = 1:numel(opts.vlines)
                    xv = datenum(opts.vlines(v));
                    yl = ax.YLim;
                    line(ax, [xv xv], yl, "Color", "k", "LineStyle", opts.vline_styles{v}, "ZData", [1 1]*ax.ZLim(2), ...
                            "HandleVisibility", "off");
                end
            end
        end

        function ax = plot_ionogram(obj, df, xparam, yparam, opts)
            arguments
                obj
                df
                xparam = "pf"
                yparam = "th"
                opts.xlabel_txt = "Frequency, MHz"
                opts.ylabel_txt = "Virtual Height, km"
                opts.ylim = [50 600]
                opts.xlim = [1 22]
                opts.xticks = [1.5 2.0 3.0 5.0 7.0 10.0 15.0 20.0]
                opts.text_txt = ""
                opts.del_ticks = false
                opts.ls = "-"
                opts.lcolor = "k"
                opts.lw = 0.7
                opts.ax = []
                opts.kind = "ionogram"
            end

            DigiPlots.setsize(obj.font_size);
            
            if ~isempty(opts.ax)
                ax = opts.ax;
            else
                ax = obj.get_axes(opts.del_ticks);
            end

            ax.XLim = log10(opts.xlim);
            ax.YLim = opts.ylim;
            if strcmpi(opts.kind, "ionogram")
                plot(ax, log10(df.(xparam)), df.(yparam), "LineStyle", opts.ls, "Color", opts.lcolor, "LineWidth", opts.lw);
            else
                scatter(ax, log10(df.(xparam)), df.(yparam), opts.lw, opts.lcolor, "s", "filled");
            end
            if ~opts.del_ticks
                ax.XTick = log10(opts.xticks);
                ax.XTickLabel = string(opts.xticks);
            end
            if strlength(opts.text_txt) > 0
                text(ax, 0.05, 0.9, opts.text_txt, "Units", "normalized", "HorizontalAlignment", "left", "VerticalAlignment", "middle", "FontSize", obj.font_size);
            end
            xlabel(ax, opts.xlabel_txt, "FontSize", obj.font_size);
            ylabel(ax, opts.ylabel_txt, "FontSize", obj.font_size);
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
