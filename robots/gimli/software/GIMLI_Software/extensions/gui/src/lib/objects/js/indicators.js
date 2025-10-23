import {Widget} from "../objects.js";
import {getColor, getFittingFontSizeSingleContainer} from "../../helpers.js";


// === BarIndicator ====================================================================================================
// export class BarIndicator extends Widget {
//     constructor(id, payload = {}) {
//         super(id, payload);
//
//         const default_config = {
//             direction: 'horizontal',  // 'horizontal' or 'vertical'
//             min_value: 0,
//             max_value: 1,
//             value: 0,
//             increment: 0.1,
//             regions: null,
//             region_colors: null,
//             bar_width: 20,  // pixels
//             bar_color: [0.9, 0.9, 0.9, 1],
//             bar_background_color: [0, 0, 0, 0],
//             bar_outline_color: [1, 1, 1, 1],
//             bar_outline_width: 1,
//             background_color: [0, 0, 0, 0],
//             title: 'Bar',
//             title_position: 'left',  // 'left' or 'top'
//             title_width: 'auto', // Can also be pixels or percentages given as strings.
//
//             show_value: true,
//             value_position: 'right',  // 'right' or 'bottom'
//             value_color: [1, 1, 1, 1],
//             value_font_size: 12,
//             value_width: 'auto',  // Can also be pixels or percentages given as strings.
//         }
//
//         this.configuration = {...this.configuration, ...default_config, ...payload.config};
//
//         this.element = this.initializeElement();
//         this.configureElement(this.element);
//         this.assignListeners(this.element);
//     }
//
//     initializeElement() {
//     }
//
//     resize() {
//     }
//
//     update(value) {
//
//     }
//
//     updateConfig(config) {
//         this.configuration = {...this.configuration, ...config};
//         this.configureElement(this.element);
//     }
// }

// === BarIndicator ====================================================================================================
export class BarIndicator extends Widget {
    constructor(id, payload = {}) {
        super(id, payload);

        const default_config = {
            // layout / data
            direction: 'horizontal',      // only horizontal implemented for now
            min_value: 0,
            max_value: 1,
            value: 0,
            increment: 0.1,
            origin: 0,                 // if null => min_value; else numeric (within [min,max] recommended)

            // ticks
            major_ticks: [0],              // numbers in same units as value
            minor_ticks: [-1, -0.5, 0.5, 1],

            // visuals
            bar_width: 10,                // px thickness of the bar (track)
            bar_color: [0.9, 0.3, 0.3, 1],
            bar_background_color: [0, 0, 0, 0],
            bar_outline_color: [1, 1, 1, 1],
            bar_outline_width: 1,         // px
            background_color: [0, 0, 0, 0],

            // title/value
            title: 'Bar',
            title_position: 'left',       // requested case
            title_width: 'auto',          // 'auto' | '80px' | '25%' ...
            show_value: true,
            value_position: 'right',      // requested case
            value_color: [1, 1, 1, 1],
            value_font_size: 12,          // px
            value_width: 'auto',          // 'auto' | size string
        };

        this.configuration = {...default_config, ...this.configuration, ...(payload?.config || {})};

        this.element = this.initializeElement();
        this.configureElement(this.element);
        this.assignListeners(this.element);

        window.addEventListener('resize', () => this.configureElement(this.element));
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // DOM
    // ────────────────────────────────────────────────────────────────────────────────
    initializeElement() {
        const el = document.createElement('div');
        el.classList.add('widget', 'barIndicator');

        // Title (left)
        this.titleEl = document.createElement('div');
        this.titleEl.classList.add('bi-title');
        el.appendChild(this.titleEl);

        // Bar container (center)
        this.barContainer = document.createElement('div');
        this.barContainer.classList.add('bi-bar-container');
        el.appendChild(this.barContainer);

        // Track (background/outline) — absolutely centered vertically
        this.barTrack = document.createElement('div');
        this.barTrack.classList.add('bi-track');

        // Ticks overlay (lines live under the fill so they only show in empty area)
        this.ticksOverlay = document.createElement('div');
        this.ticksOverlay.classList.add('bi-ticks-overlay');
        this.barTrack.appendChild(this.ticksOverlay);

        // Fill (actual value) — SQUARE corners per requirement
        this.barFill = document.createElement('div');
        this.barFill.classList.add('bi-fill');
        this.barTrack.appendChild(this.barFill);

        // Add track into container
        this.barContainer.appendChild(this.barTrack);

        // Labels row (below the bar; doesn't affect bar centering)
        this.tickLabels = document.createElement('div');
        this.tickLabels.classList.add('bi-tick-labels');
        this.barContainer.appendChild(this.tickLabels);

        // Value (right)
        this.valueEl = document.createElement('div');
        this.valueEl.classList.add('bi-value');
        el.appendChild(this.valueEl);

        return el;
    }

    getElement() {
        return this.element;
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────────────────
    _clamp(v, min, max) {
        return Math.max(min, Math.min(max, v));
    }

    _roundToIncrement(value, increment) {
        if (!increment || increment <= 0) return value;
        const rounded = Math.round(value / increment) * increment;
        return rounded;
    }

    _decimalsFromIncrement(increment) {
        if (!isFinite(increment) || increment <= 0) return 0;
        const s = increment.toString();
        if (s.includes('e-')) {
            const n = parseInt(s.split('e-')[1], 10);
            return isNaN(n) ? 0 : n;
        }
        const idx = s.indexOf('.');
        return idx >= 0 ? (s.length - idx - 1) : 0;
    }

    _valueToPct(v, min, max) {
        const range = Math.max(1e-9, max - min);
        return ((v - min) / range) * 100;
    }

    _clearTicks() {
        this.ticksOverlay.replaceChildren();
        this.tickLabels.replaceChildren();
    }

    _renderTicks() {
        const c = this.configuration;
        const {min_value, max_value} = c;

        this._clearTicks();

        const makeLabel = (v, cls, text) => {
            const pct = Math.max(0, Math.min(100, this._valueToPct(v, min_value, max_value)));
            const el = document.createElement('div');
            el.classList.add('bi-tick-label', cls);
            el.style.left = `${pct}%`;
            el.textContent = text;
            return el;
        };
        const makeLine = (v, cls) => {
            const pct = Math.max(0, Math.min(100, this._valueToPct(v, min_value, max_value)));
            const line = document.createElement('div');
            line.classList.add('bi-tick-line', cls);
            line.style.left = `${pct}%`;
            return line;
        };

        if (Array.isArray(c.minor_ticks)) {
            for (const v of c.minor_ticks) {
                if (typeof v !== 'number') continue;
                const lbl = makeLabel(v, 'minor', String(v));
                this.tickLabels.appendChild(lbl);
            }
        }

        if (Array.isArray(c.major_ticks)) {
            for (const v of c.major_ticks) {
                if (typeof v !== 'number') continue;
                const lbl = makeLabel(v, 'major', String(v));
                this.tickLabels.appendChild(lbl);

                const line = makeLine(v, 'major');
                this.ticksOverlay.appendChild(line);
            }
        }
    }

    _applyFillFromOrigin() {
        const c = this.configuration;
        const min = c.min_value;
        const max = c.max_value;

        // clamp & round value for both display and fill
        const clamped = this._clamp(c.value, min, max);
        const rounded = this._roundToIncrement(clamped, c.increment);

        // origin: null -> min; else clamp to [min,max]
        const originValue = (c.origin === null || typeof c.origin !== 'number')
            ? min
            : this._clamp(c.origin, min, max);

        const pVal = Math.max(0, Math.min(100, this._valueToPct(rounded, min, max)));
        const pOrg = Math.max(0, Math.min(100, this._valueToPct(originValue, min, max)));

        const left = Math.min(pVal, pOrg);
        const width = Math.abs(pVal - pOrg);

        this.barFill.style.left = `${left}%`;
        this.barFill.style.width = `${width}%`;

        // value label
        const decimals = this._decimalsFromIncrement(c.increment);
        if (c.show_value) this.valueEl.textContent = rounded.toFixed(decimals);
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Styling / configure
    // ────────────────────────────────────────────────────────────────────────────────
    configureElement(el) {
        super.configureElement(el);
        const c = this.configuration;

        // container bg
        el.style.backgroundColor = getColor(c.background_color);

        // grid column widths via CSS vars (keeps column alignment across stacked widgets)
        const titleW = (c.title_width === 'auto') ? 'auto' : String(c.title_width);
        const valueW = (c.value_width === 'auto') ? 'auto' : String(c.value_width);
        el.style.setProperty('--bi-title-width', titleW);
        el.style.setProperty('--bi-value-width', c.show_value ? valueW : '0px');

        // title
        this.titleEl.textContent = c.title ?? '';

        // value style (content set in _applyFillFromOrigin)
        this.valueEl.style.display = c.show_value ? '' : 'none';
        this.valueEl.style.color = getColor(c.value_color);
        this.valueEl.style.fontSize = `${c.value_font_size}px`;

        // bar sizes/colors
        const thickness = Math.max(2, Number(c.bar_width) || 20);
        const radius = Math.round(thickness * 0.4); // rounded track only
        const outlineW = Math.max(0, Number(c.bar_outline_width) || 0);

        // Track styling (centered absolutely)
        Object.assign(this.barTrack.style, {
            height: `${thickness}px`,
            borderRadius: `${radius}px`,
            backgroundColor: getColor(c.bar_background_color),
            borderColor: getColor(c.bar_outline_color),
            borderWidth: `${outlineW}px`,
            borderStyle: outlineW > 0 ? 'solid' : 'none',
        });

        // Fill styling (SQUARE corners)
        this.barFill.style.backgroundColor = getColor(c.bar_color);
        this.barFill.style.borderRadius = '0px';

        // Build ticks (labels + major lines)
        this._renderTicks();

        // Compute fill left/width based on origin
        this._applyFillFromOrigin();
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // API
    // ────────────────────────────────────────────────────────────────────────────────
    update(data) {
        // supports either {value} or raw number
        const incoming = (typeof data === 'number') ? {value: data} : (data || {});
        this.configuration = {...this.configuration, ...incoming};
        // Only the fill/label need updating in the common case:
        this._applyFillFromOrigin();
    }

    updateConfig(config) {
        this.configuration = {...this.configuration, ...config};
        this.configureElement(this.element);
    }

    resize() {
        // Re-apply for sizes/positions that depend on container.
        this.configureElement(this.element);
    }

    assignListeners(element) {
        super.assignListeners(element);
    }
}

// === CircleIndicator ================================================================================================
export class CircleIndicator extends Widget {
    constructor(id, config = {}) {
        super(id, config);

        const defaults = {
            background_color: [0, 0, 0, 1],
            color: [1, 1, 1, 0.8],
            visible: true,
            diameter: 80,            // % of container’s smaller dimension
            blinking: false,
            blinking_frequency: 1,   // Hz
        };
        this.configuration = {...defaults, ...this.configuration};

        this.element = this.initializeElement(id);

        this.configureElement(this.element);

        // recalc on window resize:
        window.addEventListener("resize", () =>
            this.configureElement(this.element)
        );
    }

    initializeElement(id) {
        const el = document.createElement("div");
        el.id = id;
        el.classList.add("widget", "circleIndicator");

        // the actual circle
        this.shapeEl = document.createElement("div");
        this.shapeEl.classList.add("circle-shape");
        el.appendChild(this.shapeEl);

        return el;
    }

    configureElement(element) {
        super.configureElement(element);
        const config = this.configuration;
        // ── Visibility & Background ───────────────────────────────────────────────
        this.element.style.display = config.visible ? "" : "none";
        this.element.style.backgroundColor = getColor(config.background_color);

        // ── Compute diameter ───────────────────────────────────────────────────────
        const rect = this.element.getBoundingClientRect();
        const size = Math.min(rect.width, rect.height);
        const diameter = size * (config.size / 100);

        // ── Style the circle ─────────────────────────────────────────────────────
        const c = this.shapeEl;
        c.style.width = `${diameter}px`;
        c.style.height = `${diameter}px`;
        c.style.borderRadius = "50%";
        c.style.backgroundColor = getColor(config.color);

        // ── Blinking ──────────────────────────────────────────────────────────────
        if (config.blinking) {
            const period = 1 / config.blinking_frequency;
            c.style.animation = `circle-blink ${period}s ease-in-out infinite`;
        } else {
            c.style.animation = "";
        }
    }

    assignListeners(element) {
        super.assignListeners(element);
    }

    getElement() {
        return this.element;
    }

    update() {
        // no dynamic data updates
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement(this.element);
    }

    resize() {
    }
}

// === LoadingIndicator ================================================================================================
export class LoadingIndicator extends Widget {
    constructor(id, config = {}) {
        super(id, config);

        const defaults = {
            background_color: [0, 0, 0, 0],
            color: [0.2, 0.2, 0.2, 1],
            thickness: 20,   // % of diameter
            size: 80,        // % of container’s smaller dimension
            speed: 1.0,      // revolutions per second
            spinning: true,
            visible: true,
        };
        this.configuration = {...defaults, ...this.configuration};

        this.element = this.initializeElement(id);
        // on resize we need to re‑compute sizes:
        window.addEventListener("resize", () =>
            this.configureElement(this.element)
        );
        this.configureElement(this.element);
    }

    initializeElement(id) {
        const el = document.createElement("div");
        el.id = id;
        el.classList.add("widget", "loadingIndicator");
        this.spinnerEl = document.createElement("div");
        this.spinnerEl.classList.add("loading-spinner");
        el.appendChild(this.spinnerEl);
        return el;
    }

    configureElement(el) {
        super.configureElement(el);
        const cfg = this.configuration;

        // show/hide & background
        el.style.display = cfg.visible ? "" : "none";
        el.style.backgroundColor = getColor(cfg.background_color);

        // figure out diameter & border thickness in px
        const rect = el.getBoundingClientRect();
        const size = Math.min(rect.width, rect.height);
        const diameter = (size * cfg.size) / 100;
        const thickness = (diameter * cfg.thickness) / 100;
        const spinnerColor = getColor(cfg.color);

        // classic hollow‑ring spinner:
        Object.assign(this.spinnerEl.style, {
            width: `${diameter}px`,
            height: `${diameter}px`,
            boxSizing: "border-box",
            border: `${thickness}px solid ${spinnerColor}`,
            borderTopColor: "transparent",         // carve out the “gap”
            borderRadius: "50%",
            background: "none",
            animation: cfg.spinning
                ? `loading-spin ${1 / cfg.speed}s linear infinite`
                : "",
        });
    }

    getElement() {
        return this.element;
    }

    assignListeners(element) { /* none */
        super.assignListeners(element);
    }

    update() { /* none */
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement(this.element);
    }

    resize() {
    }
}

// === ProgressIndicator ===============================================================================================
export class ProgressIndicator extends Widget {
    constructor(id, config = {}) {
        super(id, config);

        const defaults = {
            background_color: [0, 0, 0, 0],
            color: [0.2, 0.2, 0.2, 1],
            track_outline_color: [0.8, 0.8, 0.8, 0.2],
            track_fill_color: [0.8, 0.2, 0.2, 1],
            track_visible: true,
            type: 'linear',
            thickness: 20,               // if mode='relative' → 20% of width
            thickness_mode: 'relative',  // 'relative' or 'absolute'
            value: 0.0,
            title: '',
            title_position: 'top',       // 'top' or 'left'
            title_color: [1, 1, 1, 1],
            label: '',
            label_position: 'bottom',    // 'bottom' or 'right'
            label_color: [1, 1, 1, 1],
            ticks: [],
            tick_labels: [],
            ticks_color: [0.8, 0.8, 0.8, 1],
        };
        this.configuration = {...defaults, ...this.configuration};

        this.element = this.initializeElement(id);
        this.configureElement(this.element);

        window.addEventListener('resize', () =>
            this.configureElement(this.element)
        );
    }

    initializeElement(id) {
        const el = document.createElement('div');
        el.id = id;
        el.classList.add('widget', 'pi_widget');

        el.style.setProperty('--pi-left-width', '33%');
        el.style.setProperty('--pi-center-width', '34%');
        el.style.setProperty('--pi-right-width', '33%');
        el.style.setProperty('--pi-top-height', '33%');
        el.style.setProperty('--pi-middle-height', '34%');
        el.style.setProperty('--pi-bottom-height', '33%');

        let gridAreas = '';

        if (this.configuration.title_position === 'top') {
            gridAreas += `"title title title" `;

            if (this.configuration.label_position === 'bottom') {
                gridAreas += `"bar bar bar" `;
                gridAreas += `"label label label"`;
            } else if (this.configuration.label_position === 'right') {
                gridAreas += `"bar bar bar" `;
                gridAreas += `". . ."`;
            } else {
                console.warn(`${this.id}: Invalid label position "${this.configuration.label_position}"`);
                return null;
            }

        } else if (this.configuration.title_position === 'left') {
            if (this.configuration.label_position === 'bottom') {
                gridAreas += `". . ." `;
                gridAreas += `"title bar bar" `;
                gridAreas += `". label label"`;
            } else if (this.configuration.label_position === 'right') {
                gridAreas += `". . ." `;
                gridAreas += `"title bar label" `;
                gridAreas += `". . ."`;
            } else {
                console.warn(`${this.id}: Invalid label position "${this.configuration.label_position}"`);
                return null;
            }
        } else {
            console.warn(`${this.id}: Invalid title position "${this.configuration.title_position}"`);
            return null;
        }

        el.style.setProperty('--grid-areas', gridAreas);

        // Title
        this.titleEl = document.createElement('div');
        this.titleEl.classList.add('pi_widget_title');
        this.titleEl.textContent = 'AAAA';
        el.appendChild(this.titleEl);

        // Label
        this.labelEl = document.createElement('div');
        this.labelEl.classList.add('pi_widget_label');
        this.labelEl.textContent = 'BBBB';
        el.appendChild(this.labelEl);

        // Bar container
        this.barContainer = document.createElement('div');
        this.barContainer.classList.add('pi_widget_bar_container');
        el.appendChild(this.barContainer);

        // Outline (track)
        this.barOutline = document.createElement('div');
        this.barOutline.classList.add('pi-bar-outline');
        this.barContainer.appendChild(this.barOutline);

        // Fill (progress)
        this.barFill = document.createElement('div');
        this.barFill.classList.add('pi-bar-fill');
        this.barOutline.appendChild(this.barFill);

        return el;
    }

    configureElement(el) {
        super.configureElement(el);
        const c = this.configuration;

        // title
        this.titleEl.textContent = c.title;
        this.titleEl.style.color = getColor(c.title_color);
        el.dataset.titlePosition = c.title_position;


        // Label
        this.labelEl.textContent = c.label;
        this.labelEl.style.color = getColor(c.label_color);
        el.dataset.labelPosition = c.label_position;

        // bar thickness & rounding
        const rect = this.barContainer.getBoundingClientRect();
        let thicknessPx;
        if (c.thickness_mode === 'absolute') {
            thicknessPx = c.thickness;
        } else {
            thicknessPx = (rect.width * c.thickness) / 100;
        }
        const radius = thicknessPx * 0.25;  // gentler rounding

        this.barOutline.style.height = `${thicknessPx}px`;
        this.barOutline.style.borderRadius = `${radius}px`;
        this.barOutline.style.borderWidth = '1px';
        this.barOutline.style.borderStyle = 'solid';
        this.barOutline.style.borderColor = getColor(c.track_outline_color);

        this.barFill.style.height = '100%';
        this.barFill.style.borderRadius = `${radius}px`;
        this.barFill.style.backgroundColor = getColor(c.track_fill_color);

        // fill %
        const pct = Math.max(0, Math.min(100, c.value * 100));
        this.barFill.style.width = `${pct}%`;


        this.showBar(this.configuration.track_visible);

    }

    assignListeners() { /* none */
    }

    getElement() {
        return this.element;
    }

    update(data) {
        const value = data.value || this.configuration.value;
        const label = data.label || this.configuration.label;

        // Update the value and label in the configuration
        this.configuration.value = value;
        this.configuration.label = label;

        // Change the fill width based on the new value
        const pct = Math.max(0, Math.min(100, value * 100));
        this.barFill.style.width = `${pct}%`;

        // Change the label text
        this.labelEl.textContent = label;
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement(this.element);
    }

    showBar(show) {
        this.barContainer.style.display = show ? 'block' : 'none';
    }

    resize() {
    }
}


// === BatteryIndicator ==================================================================================================
export class BatteryIndicatorWidget extends Widget {
    constructor(id, payload = {}) {
        super(id, payload);
        const defaults = {
            show: 'percentage',  // 'percentage', 'voltage' or null
            label_position: 'right', // 'left', 'center', 'right'
            label_color: [1, 1, 1, 1],
            thresholds: {low: 0.2, medium: 0.7},
            value: 0.6,
            voltage: 0.1
        };
        this.configuration = {...defaults, ...this.configuration};
        this.element = this._initializeElement();
        this.configureElement(this.element);
        this.assignListeners(this.element);
    }

    _initializeElement() {
        const el = document.createElement('div');
        el.classList.add('widget', 'battery-indicator');
        // battery body
        this.bodyEl = document.createElement('div');
        this.bodyEl.classList.add('battery-body');
        // battery head
        this.headEl = document.createElement('div');
        this.headEl.classList.add('battery-head');
        el.appendChild(this.bodyEl);
        el.appendChild(this.headEl);
        // fill
        this.fillEl = document.createElement('div');
        this.fillEl.classList.add('battery-fill');
        this.bodyEl.appendChild(this.fillEl);
        // label
        this.labelEl = document.createElement('div');
        this.labelEl.classList.add('battery-label');
        el.appendChild(this.labelEl);
        return el;
    }


    configureElement() {
        super.configureElement(this.element);
        const cfg = this.configuration;
        const pct = Math.max(0, Math.min(1, cfg.value));
        const widthPct = pct * 100;

        // — fill bar —
        let fillColor;
        if (pct <= cfg.thresholds.low) fillColor = 'darkred';
        else if (pct <= cfg.thresholds.medium) fillColor = 'darkorange';
        else fillColor = 'darkgreen';

        this.fillEl.style.width = `${widthPct}%`;
        this.fillEl.style.backgroundColor = fillColor;
        this.fillEl.style.position = 'absolute';
        this.fillEl.style.top = '0';
        this.fillEl.style.left = '0';
        this.fillEl.style.zIndex = '1';

        // — label text & base styles —
        let text = '';
        if (cfg.show === 'percentage') text = `${Math.round(widthPct)}%`;
        else if (cfg.show === 'voltage') text = `${cfg.voltage.toFixed(1)}V`;
        this.labelEl.textContent = text;
        this.labelEl.style.color = getColor(cfg.label_color);
        this.element.dataset.labelPosition = cfg.label_position;

        // — reposition the label —
        if (cfg.label_position === 'center') {
            // make sure body is the positioning context
            this.bodyEl.style.position = 'relative';

            // move label _into_ the battery-body
            this.bodyEl.appendChild(this.labelEl);

            Object.assign(this.labelEl.style, {
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                margin: '0',
                whiteSpace: 'nowrap',
                zIndex: '2',
                pointerEvents: 'none',
            });
        } else {
            // move label back to root, clear absolute styles
            this.element.appendChild(this.labelEl);
            this.labelEl.style.position = '';
            this.labelEl.style.top = '';
            this.labelEl.style.left = '';
            this.labelEl.style.transform = '';
            this.labelEl.style.zIndex = '';
            // add some margin
            if (cfg.label_position === 'left') {
                this.labelEl.style.margin = '0 8px 0 0';
            } else {  // right
                this.labelEl.style.margin = '0 0 0 8px';
            }
        }
    }

    setValue(percentage, voltage) {
        this.configuration.value = percentage;
        this.configuration.voltage = voltage;
        this.configureElement();
    }

    getElement() {
        return this.element;
    }

    assignListeners(element) {
        super.assignListeners(element);
    }

    initializeElement() {
    }

    update(data) {
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement();
    }

    resize() {
    }
}

// === ConnectionIndicator ================================================================================================
export class ConnectionIndicator extends Widget {
    constructor(id, config = {}) {
        super(id, config);
        const defaults = {
            color: [0.8, 0.8, 0.8, 1],
            value: 'medium', // 'low','medium','high'
        };
        this.configuration = {...defaults, ...this.configuration};
        this.element = this._initializeElement();
        this.configureElement(this.element);
        this.assignListeners(this.element);
    }

    _initializeElement() {
        const el = document.createElement('div');
        el.classList.add('widget', 'connection-indicator');
        this.bars = [];
        for (let i = 1; i <= 3; i++) {
            const bar = document.createElement('div');
            bar.classList.add('connection-bar');
            bar.dataset.level = i;
            el.appendChild(bar);
            this.bars.push(bar);
        }
        return el;
    }

    configureElement() {
        super.configureElement(this.element);
        const cfg = this.configuration;
        const levelMap = {low: 1, medium: 2, high: 3};
        const filledCount = levelMap[cfg.value] || 0;
        const color = getColor(cfg.color);
        this.bars.forEach((bar, idx) => {
            if (idx < filledCount) {
                bar.style.backgroundColor = color;
                bar.style.borderColor = color;
                bar.style.opacity = '1';
            } else {
                bar.style.backgroundColor = 'transparent';
                bar.style.borderColor = color;
                bar.style.opacity = '1';
            }
        });
    }

    setValue(value) {
        this.configuration.value = value;
        this.configureElement();
    }

    getElement() {
        return this.element;
    }

    assignListeners(element) {
        super.assignListeners(element);
    }

    initializeElement() {
    }

    resize() {
    }

    update(data) {
        return undefined;
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement();
    }
}

// === InternetIndicator ================================================================================================
export class InternetIndicator extends Widget {
    constructor(id, config = {}) {
        super(id, config);
        const defaults = {available: true, fit_to_container: true};
        this.configuration = {...defaults, ...this.configuration};
        this.element = this._initializeElement();
        this.configureElement(this.element);
        this.assignListeners(this.element);
    }

    _initializeElement() {
        const el = document.createElement('div');
        el.classList.add('widget', 'internet-indicator');
        this.iconEl = document.createElement('div');
        this.iconEl.classList.add('internet-icon');
        this.iconEl.textContent = "🌍";
        el.appendChild(this.iconEl);
        this.crossEl = document.createElement('div');
        this.crossEl.classList.add('internet-cross');
        el.appendChild(this.crossEl);
        return el;
    }

    configureElement() {
        super.configureElement(this.element);
        const cfg = this.configuration;
        if (cfg.available) {
            this.iconEl.style.opacity = '1';
            this.crossEl.style.display = 'none';
        } else {
            this.iconEl.style.opacity = '0.4';
            this.crossEl.style.display = 'block';
        }
    }

    setValue(available) {
        this.configuration.available = available;
        this.configureElement();
    }

    assignListeners(element) {
        super.assignListeners(element);
    }

    getElement() {
        return this.element;
    }

    initializeElement() {
    }

    update(data) {
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement();
    }

    resize() {
        if (this.configuration.fit_to_container) {
            getFittingFontSizeSingleContainer(this.iconEl, 0, 0, 100, 0);
        }
    }
}

// === InternetIndicator ================================================================================================
export class NetworkIndicator extends Widget {
    constructor(id, config = {}) {
        super(id, config);
        const defaults = {available: true};
        this.configuration = {...defaults, ...this.configuration};
        this.element = this._initializeElement();
        this.configureElement(this.element);
        this.assignListeners(this.element);
    }

    _initializeElement() {
        const el = document.createElement('div');
        el.classList.add('widget', 'network-indicator');
        this.iconEl = document.createElement('div');
        this.iconEl.classList.add('network-icon');
        el.appendChild(this.iconEl);
        this.crossEl = document.createElement('div');
        this.crossEl.classList.add('network-cross');
        el.appendChild(this.crossEl);
        return el;
    }

    configureElement() {
        super.configureElement(this.element);
        const cfg = this.configuration;
        if (cfg.available) {
            this.iconEl.style.opacity = '1';
            this.crossEl.style.display = 'none';
        } else {
            this.iconEl.style.opacity = '0.4';
            this.crossEl.style.display = 'block';
        }
    }

    setValue(available) {
        this.configuration.available = available;
        this.configureElement();
    }

    assignListeners(element) {
        super.assignListeners(element);
    }

    getElement() {
        return this.element;
    }

    initializeElement() {
    }

    update(data) {
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement();
    }

    resize() {
    }
}


// === JoystickIndicator ================================================================================================
export class JoystickIndicator extends Widget {
    constructor(id, config = {}) {
        super(id, config);
        // add png_icon_path so it can be overridden if needed
        const defaults = {
            available: true,
            use_png_icon: true,
            png_icon_path: '/gamepad.png'
        };
        this.configuration = {...defaults, ...this.configuration};
        this.element = this._initializeElement();
        this.configureElement();
        this.assignListeners(this.element);
    }

    _initializeElement() {
        const el = document.createElement('div');
        el.classList.add('widget', 'highlightable', 'joystick-indicator');

        // choose between PNG or emoji
        if (this.configuration.use_png_icon) {
            this.iconEl = document.createElement('img');
            this.iconEl.src = this.configuration.png_icon_path;
            this.iconEl.alt = 'Gamepad';
            this.iconEl.classList.add('joystick-icon-image');
        } else {
            this.iconEl = document.createElement('div');
            this.iconEl.classList.add('joystick-icon');
        }
        el.appendChild(this.iconEl);

        this.crossEl = document.createElement('div');
        this.crossEl.classList.add('joystick-cross');
        el.appendChild(this.crossEl);

        return el;
    }

    configureElement() {
        super.configureElement(this.element);
        const {available} = this.configuration;
        if (available) {
            this.iconEl.style.opacity = '1';
            this.crossEl.style.display = 'none';
        } else {
            this.iconEl.style.opacity = '0.4';
            this.crossEl.style.display = 'block';
        }
    }

    setValue(available) {
        this.configuration.available = available;
        this.configureElement();
    }

    assignListeners(element) {
        super.assignListeners(element);

        // Assign click listener to the joystick icon
        this.iconEl.addEventListener('click', () => {
            // Trigger the click on the parent element
            this.callbacks.get('event').call({id: this.id, event: 'click', data: {}});
        });
    }

    getElement() {
        return this.element;
    }

    initializeElement() {
    }

    update(data) {
    }

    updateConfig(data) {
        this.configuration = {...this.configuration, ...data};
        this.configureElement();
    }

    resize() {
    }
}