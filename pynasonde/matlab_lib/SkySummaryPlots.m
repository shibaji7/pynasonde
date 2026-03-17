classdef SkySummaryPlots < DigiPlots
    %SKYSUMMARYPLOTS Polar/sky plotting helpers and drift plots.

    methods
        function obj = SkySummaryPlots(fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw, draw_local_time)
            arguments
                fig_title = "" 
                nrows = 1 
                ncols = 1
                font_size = 16
                figsize = [3 3]
                date = []
                date_lims = []
                subplot_kw = struct("projection", "polar")
                draw_local_time = false
            end
            obj@DigiPlots(fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw, draw_local_time);
        end

        function plot_skymap(obj, df, xparam, yparam, zparam, opts)
            arguments
                obj
                df
                xparam = "x_coord"
                yparam = "y_coord"
                zparam = "spect_dop_freq"
                opts.theta_lim = [0 360]
                opts.rlim = 21
                opts.text_txt = ""
                opts.del_ticks = true
                opts.cmap = "redblackblue"
                opts.cbar = true
                opts.clim = [-1 1]
                opts.cbar_label = "Doppler, Hz"
                opts.ms = 90
                opts.zorder = 2
                opts.nrticks = 5
                opts.txt_loc = [0.05 0.9]
                opts.tag_direction = false
            end

            DigiPlots.setsize(obj.font_size);
            ax = obj.get_axes(opts.del_ticks);

            x = df.(xparam);
            y = df.(yparam);
            z = df.(zparam);
            r = sqrt(x.^2 + y.^2);
            theta = -atan2(y, x);

            cmap_arr = DigiPlots.resolve_colormap(opts.cmap);
            disp("cmap: " + string(opts.cmap))
            colormap(ax, cmap_arr);
            if isa(ax, "matlab.graphics.axis.PolarAxes") 
                im = polarscatter(ax, theta, r, opts.ms, z, "filled"); 
            else
                im = scatter(ax, theta, r, opts.ms, z, "filled");
            end
            caxis(ax, opts.clim);
            ax.Box = "off";

            hold(ax, "on");
            for rtick = linspace(0, opts.rlim - 1, opts.nrticks)
                th = linspace(0, 2*pi, 200); 
                polarplot(ax, th, rtick * ones(size(th)), "--", "Color", [0 0 0], "LineWidth", 0.4); 
            end 
            for th = [0 pi/2 pi 1.5*pi] 
                polarplot(ax, [th th], [0 opts.rlim], "-", "Color", [0 0 0], "LineWidth", 0.4);
            end

            if opts.tag_direction
                text(ax, pi/2, opts.rlim * 1.05, "East", "HorizontalAlignment", "left", "VerticalAlignment", "middle",  "Rotation", 90); 
                text(ax, 0, opts.rlim * 1.05, "North", "HorizontalAlignment", "center", "VerticalAlignment", "bottom"); 
            end

            if strlength(opts.text_txt) > 0 
                text(ax, opts.txt_loc(1), opts.txt_loc(2), opts.text_txt, "Units", "normalized", "HorizontalAlignment", "left", "VerticalAlignment", "middle"); 
            end 

            if opts.cbar
                obj.add_colorbar(im, ax, opts.cbar_label, [-0.01 0 0.2 0.5]);
            end

            ax.ThetaLim = opts.theta_lim;
            ax.RLim = [0 opts.rlim];
            ax.ThetaTick = []; 
            ax.RTick = []; 
            ax.ThetaZeroLocation = "top"; 
            ax.ThetaDir = "clockwise";
        end

        function plot_drift_velocities(obj, df, xparam, yparam, color, error, ylabel_txt, xlabel_txt, text_txt, del_ticks, fmt, lw, alpha, zorder, major_locator, minor_locator, ylim, xlim)
            if nargin < 3, xparam = "datetime"; end
            if nargin < 4, yparam = "Vx"; end
            if nargin < 5, color = "r"; end
            if nargin < 6, error = "Vx_err"; end
            if nargin < 7, ylabel_txt = "Velocity (V_x), m/s"; end
            if nargin < 8, xlabel_txt = "Time, UT"; end
            if nargin < 9, text_txt = ""; end
            if nargin < 10, del_ticks = false; end
            if nargin < 11, fmt = "o"; end
            if nargin < 12, lw = 0.7; end
            if nargin < 13, alpha = 0.8; end
            if nargin < 14, zorder = 4; end
            if nargin < 15, major_locator = []; end
            if nargin < 16, minor_locator = []; end
            if nargin < 17, ylim = [-100 100]; end
            if nargin < 18, xlim = []; end
            %#ok<NASGU>

            DigiPlots.setsize(obj.font_size);
            ax = obj.get_axes(del_ticks);

            xdata = df.(xparam);
            if isempty(xlim)
                xlim = [min(xdata) max(xdata)];
            end
            ax.XLim = datenum(xlim);
            xlabel(ax, xlabel_txt);
            ax.YLim = ylim;
            ylabel(ax, ylabel_txt);

            eb = errorbar(ax, df.(xparam), df.(yparam), df.(error), fmt, ...
                "Color", color, "LineWidth", lw, "CapSize", 1);
            % eb.MarkerFaceAlpha = alpha;
            if strlength(text_txt) > 0
                text(ax, 0.05, 0.9, text_txt, "Units", "normalized", "HorizontalAlignment", "left", "VerticalAlignment", "middle", "FontSize", obj.font_size);
            end
            if nargin >= 14 %#ok<*INUSD>
                % keep MATLAB from optimizing away zorder
            end
        end
    end

    methods (Static)
        function dvlplot = plot_dvl_drift_velocities(df, xparam, yparams, colors, errors, labels, text_txt, del_ticks, fmt, lw, alpha, zorder, major_locator, minor_locator, ylim, xlim, fname, figsize, draw_local_time, font_size)
            if nargin < 2, xparam = "datetime"; end
            if nargin < 3, yparams = {"Vx", "Vy", "Vz"}; end
            if nargin < 4, colors = {"r", "b", "k"}; end
            if nargin < 5, errors = {"Vx_err", "Vy_err", "Vz_err"}; end
            if nargin < 6, labels = {"$V_x$", "$V_y$", "$V_z$"}; end
            if nargin < 7, text_txt = ""; end
            if nargin < 8, del_ticks = false; end
            if nargin < 9, fmt = "o"; end
            if nargin < 10, lw = 0.7; end
            if nargin < 11, alpha = 0.8; end
            if nargin < 12, zorder = 4; end
            if nargin < 13, major_locator = []; end
            if nargin < 14, minor_locator = []; end
            if nargin < 15, ylim = [-100 100]; end
            if nargin < 16, xlim = []; end
            if nargin < 17, fname = ""; end
            if nargin < 18, figsize = [2.5 7]; end
            if nargin < 19, draw_local_time = false; end
            if nargin < 20, font_size = 18; end
            %#ok<NASGU>

            if draw_local_time
                local_name = "local_" + string(xparam);
                if ismember(local_name, df.Properties.VariableNames)
                    xparam = char(local_name);
                end
            end

            dvlplot = SkySummaryPlots("", 3, 1, font_size, figsize, [], [], struct(), draw_local_time);
            for i = 1:numel(yparams)
                y = yparams{i};
                col = colors{i};
                err = errors{i};
                lab = labels{i};

                text_i = text_txt;
                if i == 1
                    d0 = df.(xparam)(1);
                    d1 = df.(xparam)(end);
                    if isdatetime(d0) && isdatetime(d1)
                        if dateshift(d0, "start", "day") ~= dateshift(d1, "start", "day")
                            date_txt = sprintf("%s-%s", datestr(d0, "dd mmm"), datestr(d1, "dd mmm, yyyy"));
                        else
                            date_txt = datestr(d1, "dd mmm, yyyy");
                        end
                        text_i = text_i + "" + date_txt;
                    end
                else
                    text_i = "";
                end

                ylabel = "Velocity(" + lab + "), m/s";
                xlabel = "";
                if i == numel(yparams)
                    xlabel = "Time, UT";
                    if draw_local_time
                        xlabel = "Time, LT";
                    end
                end

                dvlplot.plot_drift_velocities(df, xparam, y, col, err, ylabel, xlabel, text_i, del_ticks, fmt, lw, alpha, zorder, major_locator, minor_locator, ylim, xlim);
            end

            if strlength(fname) > 0
                dvlplot.save(fname);
                dvlplot.close();
                dvlplot = [];
            end
        end
    end
end
