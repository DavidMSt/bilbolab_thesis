import {Widget} from "../objects.js";
import {getColor, shadeColor, interpolateColors} from "../../helpers.js";

export class TableWidget extends Widget {
    constructor(id, config = {}) {
        super(id, config);

        // Default settings mirroring Python defaults
        const defaults = {
            columns: [],                // array of { id, title, width, font_size, background_color, header_background_color, header_text_color, text_align, font_weight, font_style, font_family, type, number_increment }
            rows: [],                   // array of { row_id, index, cells: [...], font_size, background_color, text_align, text_color, font_weight, font_style, font_family }
            has_header: true,
            vertical_fit: false,
            line_width: 1,
            line_color: '#ccc',
            background_color: 'transparent',
            scrollbar_track_color: 'transparent',
            scrollbar_thumb_color: [0.4, 0.4, 0.4],
            font_size: 10,              // pt
            text_color: '#000',
            header_background_color: 'transparent',
            header_font_size: 12,       // pt
            header_font_weight: 'bold',
            header_font_style: 'normal',
            header_text_color: '#000',
        };

        this.configuration = {...defaults, ...this.configuration};
        this._normalizeColumnsAndRows();

        this.element = document.createElement('div');
        this.element.id = this.id;
        this.element.classList.add('widget', 'tableWidget');

        this.table = document.createElement('table');
        this.table.style.width = '100%';
        this.table.style.borderCollapse = 'collapse';
        this.element.appendChild(this.table);

        this.thead = document.createElement('thead');
        this.tbody = document.createElement('tbody');
        this.table.appendChild(this.thead);
        this.table.appendChild(this.tbody);

        this._prevInputs = {};   // keep track of last “committed” value per cell
        this._tooltipEl = null; // single floating tooltip element
        this._tooltipTimeout = null;

        this.configureElement(this.element);
        this.assignListeners(this.element);
    }

    /**
     * Normalize columns to an array, and for each row
     * build a quick map of cell overrides by column_id.
     */
    _normalizeColumnsAndRows() {
        const c = this.configuration;
        if (c.columns && !Array.isArray(c.columns)) {
            c.columns = Object.values(c.columns);
        }
        if (c.rows && !Array.isArray(c.rows)) {
            c.rows = Object.values(c.rows);
        }
        // sort rows by index
        c.rows.sort((a, b) => (a.index || 0) - (b.index || 0));
        // build a map row._cellMap for each row
        c.rows.forEach(row => {
            row._cellMap = {};
            if (Array.isArray(row.cells)) {
                row.cells.forEach(cell => {
                    if (cell.column_id) row._cellMap[cell.column_id] = cell;
                });
            }
        });
    }

    configureElement(element) {
        super.configureElement(element);
        // this.configuration = {...this.configuration, ...config};
        this._normalizeColumnsAndRows();

        // Container styling
        const c = this.configuration;
        this.element.style.backgroundColor = getColor(c.background_color);
        this.element.style.overflowY = c.vertical_fit ? 'hidden' : 'auto';

        const track = getColor(c.scrollbar_track_color || c.background_color);
        const thumb = getColor(c.scrollbar_thumb_color || shadeColor(c.background_color, -10));
        this.element.style.setProperty('--scrollbar-track', track);
        this.element.style.setProperty('--scrollbar-thumb', thumb);

        this._buildHeader();
        this._buildBody();
    }

    getElement() {
        return this.element;
    }

    _buildHeader() {
        const c = this.configuration;
        this.thead.innerHTML = '';
        if (!c.has_header) return;

        const tr = document.createElement('tr');
        c.columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.title || col.column_id;

            if (col.width != null) {
                // if width is a number, treat it as percentage
                th.style.width = typeof col.width === 'number'
                    ? `${col.width * 100}%`
                    : col.width;
            }

            th.style.fontSize = (col.header_font_size ?? c.header_font_size) + 'pt';
            th.style.fontWeight = col.header_font_weight ?? c.header_font_weight;
            th.style.fontStyle = col.header_font_style ?? c.header_font_style;
            th.style.backgroundColor = getColor(col.header_background_color ?? c.header_background_color);
            th.style.color = getColor(col.header_text_color ?? c.header_text_color);
            th.style.textAlign = 'center';

            if (c.line_width > 0) {
                th.style.border = `${c.line_width}px solid ${getColor(c.line_color)}`;
            } else {
                th.style.border = 'none';
            }

            tr.appendChild(th);
        });
        this.thead.appendChild(tr);
    }

    _buildBody() {
        const c = this.configuration;
        this.tbody.innerHTML = '';

        c.rows.forEach((row, rowIndex) => {
            const tr = document.createElement('tr');
            tr.dataset.rowId = row.row_id ?? rowIndex;

            c.columns.forEach(col => {
                const td = document.createElement('td');
                const cellData = row._cellMap[col.column_id] || null;
                this.configureCell(td, row, col, cellData);
                tr.appendChild(td);
            });

            this.tbody.appendChild(tr);
        });
    }

    /**
     * Apply value + all style overrides in the order:
     * cell > row > column > table
     */
    configureCell(td, row, col, cellData) {
        const c = this.configuration;

        // determine raw value
        let rawValue = (cellData && cellData.value != null)
            ? cellData.value
            : (row[col.column_id] != null ? row[col.column_id] : '');

        // clear any existing content
        td.innerHTML = '';

        // format & render by type
        switch (col.type) {
            case 'number': {
                // determine decimals from number_increment
                let decimals = 0;
                if (col.number_increment != null) {
                    const s = col.number_increment.toString();
                    decimals = (s.includes('.') ? s.split('.')[1].length : 0);
                }
                const num = parseFloat(rawValue);
                const text = isNaN(num) ? '' : num.toFixed(decimals);
                td.textContent = text;
                td.style.fontFamily = 'monospace';
                break;
            }
            case 'date': {
                const d = new Date(rawValue);
                if (!isNaN(d)) {
                    const y = d.getFullYear();
                    const m = String(d.getMonth() + 1).padStart(2, '0');
                    const day = String(d.getDate()).padStart(2, '0');
                    td.textContent = `${y}-${m}-${day}`;
                }
                break;
            }
            case 'datetime': {
                const d2 = new Date(rawValue);
                if (!isNaN(d2)) {
                    const y2 = d2.getFullYear();
                    const m2 = String(d2.getMonth() + 1).padStart(2, '0');
                    const day2 = String(d2.getDate()).padStart(2, '0');
                    const h = String(d2.getHours()).padStart(2, '0');
                    const min = String(d2.getMinutes()).padStart(2, '0');
                    const s = String(d2.getSeconds()).padStart(2, '0');
                    td.textContent = `${y2}-${m2}-${day2} ${h}:${min}:${s}`;
                }
                break;
            }
            case 'boolean': {
                td.textContent = rawValue ? 'true' : 'false';
                break;
            }
            case 'select': {
                const sel = document.createElement('select');
                sel.classList.add('tableSelect');

                const bg = (cellData && cellData.background_color)
                    || row.background_color
                    || col.background_color
                    || c.background_color
                    || [0.15, 0.15, 0.15];
                const select_color = interpolateColors(bg, [1, 1, 1], 0.3)
                const tc = (cellData && cellData.text_color)
                    || row.text_color
                    || col.text_color
                    || c.text_color;

                // 2) apply them to the select
                sel.style.backgroundColor = getColor(select_color);
                sel.style.color = getColor(tc);


                sel.style.margin = '0';
                // sel.style.padding = '2px 4px';     // tweak as needed
                sel.style.boxSizing = 'border-box';
                sel.style.height = '100%';         // fill vertical space
                sel.style.width = '100%';

                const disabled = (cellData && cellData.disabled) || col.disabled || false;

                if (disabled) {
                    sel.disabled = true;
                    sel.classList.add('tableSelect--disabled');
                }


                const options = (cellData && cellData.select_options) || [];
                options.forEach(optVal => {
                    const opt = document.createElement('option');
                    opt.value = optVal;
                    opt.textContent = optVal;
                    if (optVal === rawValue) opt.selected = true;
                    sel.appendChild(opt);
                });
                sel.addEventListener('change', () => {
                    this._onSelectEvent({
                        row_id: row.row_id,
                        column_id: col.column_id,
                        value: sel.value
                    });
                });
                td.appendChild(sel);
                break;
            }
            case 'input': {
                const inp = document.createElement('input');
                inp.classList.add('tableInput');
                inp.type = 'text';
                inp.value = rawValue != null ? rawValue : '';
                inp.style.width = '100%';
                inp.style.height = '100%';

                // ← hook up our “prev” store
                const key = `${row.row_id}_${col.column_id}`;
                this._prevInputs[key] = inp.value;

                const commit = () => {
                    const v = inp.value;
                    this._onInputEvent({
                        row_id: row.row_id,
                        column_id: col.column_id,
                        value: v
                    });
                };

                inp.addEventListener('keydown', e => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        commit();
                        inp.blur();
                    }
                });

                inp.addEventListener('blur', () => {
                    // revert to last-accepted value
                    inp.value = this._prevInputs[key];
                    this._hideCellTooltip();
                });

                td.appendChild(inp);
                break;
            }


            case 'checkbox': {
                const cb = document.createElement('input');
                cb.classList.add('tableCheckbox');
                cb.type = 'checkbox';

                const disabled = (cellData && cellData.disabled) || col.disabled || false;

                if (disabled) {
                    cb.disabled = true;
                    cb.classList.add('tableCheckbox--disabled');
                }

                cb.checked = !!rawValue;
                cb.addEventListener('change', () => {
                    this._onCheckboxEvent({
                        row_id: row.row_id,
                        column_id: col.column_id,
                        value: cb.checked ? 1 : 0
                    });
                });
                td.appendChild(cb);
                break;
            }
            case 'button': {
                const btn = document.createElement('button');
                btn.classList.add('tableButton');
                btn.textContent = rawValue != null ? String(rawValue) : '';
                btn.style.width = '100%';

                // 1) compute the same bg & text colors
                const bg = (cellData && cellData.button_color)
                    || row.button_color
                    || col.button_color
                    || c.button_color
                    || [0.15, 0.15, 0.15];
                const tc = (cellData && cellData.text_color)
                    || row.text_color
                    || col.text_color
                    || c.text_color;

                // 2) apply them to the button
                btn.style.backgroundColor = getColor(bg);
                btn.style.color = getColor(tc);

                // 3) remove margins & shrink padding
                btn.style.margin = '0';
                btn.style.padding = '2px 4px';     // tweak as needed
                btn.style.boxSizing = 'border-box';
                btn.style.height = '100%';         // fill vertical space
                btn.style.width = '100%';

                const disabled = (cellData && cellData.disabled) || col.disabled || false;

                if (disabled) {
                    btn.disabled = true;
                    btn.classList.add('tableButton--disabled');
                }

                btn.addEventListener('click', () => {
                    this._onButtonEvent({
                        row_id: row.row_id,
                        column_id: col.column_id,
                        value: rawValue
                    });
                });
                td.appendChild(btn);
                break;
            }

            case 'text':
            default: {
                td.textContent = rawValue;
            }
        }

        // COMMON STYLING (background, color, font, align, borders)
        // background
        const bg = (cellData && cellData.background_color)
            || row.background_color
            || col.background_color
            || c.background_color;
        td.style.backgroundColor = getColor(bg);

        // text color
        const tc = (cellData && cellData.text_color)
            || row.text_color
            || col.text_color
            || c.text_color;
        td.style.color = getColor(tc);

        // font size
        const fs = (cellData && cellData.font_size)
            || row.font_size
            || col.font_size
            || c.font_size;
        td.style.fontSize = fs + 'pt';

        // text align
        td.style.textAlign = (cellData && cellData.text_align)
            || row.text_align
            || col.text_align
            || 'center';

        // font weight & style & family (except the monospace override above)
        if (cellData && cellData.font_weight) td.style.fontWeight = cellData.font_weight;
        else if (row.font_weight) td.style.fontWeight = row.font_weight;
        else if (col.font_weight) td.style.fontWeight = col.font_weight;

        if (cellData && cellData.font_style) td.style.fontStyle = cellData.font_style;
        else if (row.font_style) td.style.fontStyle = row.font_style;
        else if (col.font_style) td.style.fontStyle = col.font_style;

        if (cellData && cellData.font_family) td.style.fontFamily = cellData.font_family;
        else if (row.font_family) td.style.fontFamily = row.font_family;
        else if (col.font_family) td.style.fontFamily = col.font_family;

        // borders
        if (c.line_width > 0) {
            td.style.border = `${c.line_width}px solid ${getColor(c.line_color)}`;
        } else {
            td.style.border = 'none';
        }
    }

    updateConfig(data) {
        if ('columns' in data || 'rows' in data || 'has_header' in data) {
            this.configureElement(this.element);
            return;
        }
        if (data.addedRow) this._onAddRow(data.addedRow);
        if (data.removedRow) this._onRemoveRow(data.removedRow);
        if (data.updatedRow) this._onUpdateRow(data.updatedRow);
        if (data.addedColumn) this._onAddColumn(data.addedColumn);
        if (data.removedColumn) this._onRemoveColumn(data.removedColumn);
        if (data.updatedCell) this._onUpdateCell(data.updatedCell);
    }

    update(data) {
        const update_type = data.type;

        switch (update_type) {
            case 'cell_change':
                this._onUpdateCell(data.data);
                break;
            default:
                console.warn(`TableWidget: Unsupported update type "${update_type}"`);
                break;
        }
    }

    _onAddRow({row, index}) {
        const c = this.configuration;
        const idx = (index == null || index > c.rows.length) ? c.rows.length : index;
        if (!row.row_id) row.row_id = `row_${Date.now()}`;

        row._cellMap = {};
        if (row.cells) row.cells.forEach(cell => row._cellMap[cell.column_id] = cell);

        c.rows.splice(idx, 0, row);

        const tr = document.createElement('tr');
        tr.dataset.rowId = row.row_id;
        c.columns.forEach(col => {
            const td = document.createElement('td');
            this.configureCell(td, row, col, row._cellMap[col.column_id] || null);
            tr.appendChild(td);
        });

        const before = this.tbody.children[idx];
        if (before) this.tbody.insertBefore(tr, before);
        else this.tbody.appendChild(tr);
    }

    _onRemoveRow(arg) {
        const c = this.configuration;
        const idx = typeof arg === 'number'
            ? arg
            : c.rows.findIndex(r => r.row_id === arg);
        if (idx < 0 || idx >= c.rows.length) return;

        c.rows.splice(idx, 1);
        const tr = this.tbody.children[idx];
        if (tr) this.tbody.removeChild(tr);
    }

    _onUpdateRow({rowId, row}) {
        const c = this.configuration;
        const idx = typeof rowId === 'number'
            ? rowId
            : c.rows.findIndex(r => r.row_id === rowId);
        if (idx < 0) return;

        row._cellMap = {};
        if (row.cells) row.cells.forEach(cell => row._cellMap[cell.column_id] = cell);

        c.rows[idx] = row;

        const oldTr = this.tbody.children[idx];
        const newTr = document.createElement('tr');
        newTr.dataset.rowId = row.row_id;
        c.columns.forEach(col => {
            const td = document.createElement('td');
            this.configureCell(td, row, col, row._cellMap[col.column_id] || null);
            newTr.appendChild(td);
        });
        if (oldTr) this.tbody.replaceChild(newTr, oldTr);
    }

    _onUpdateCell(payload) {
        const rowId = payload.rowId ?? payload.row_id;
        const columnId = payload.columnId ?? payload.column_id;
        const {value, ...overrides} = payload;

        const c = this.configuration;
        const rowIdx = typeof rowId === 'number'
            ? rowId
            : c.rows.findIndex(r => r.row_id === rowId);
        const colIdx = c.columns.findIndex(c => c.column_id === columnId);
        if (rowIdx < 0 || colIdx < 0) return;

        const row = c.rows[rowIdx];
        row._cellMap[columnId] = {...(row._cellMap[columnId] || {}), value, ...overrides};
        row[columnId] = value;

        const tr = this.tbody.children[rowIdx];
        if (!tr) return;
        const td = tr.children[colIdx];
        if (!td) return;
        this.configureCell(td, row, c.columns[colIdx], row._cellMap[columnId]);
    }

    _onAddColumn({column, index}) {
        const c = this.configuration;
        const idx = (index == null || index > c.columns.length)
            ? c.columns.length : index;
        if (!column.column_id) column.column_id = `col_${Date.now()}`;
        c.columns.splice(idx, 0, column);

        c.rows.forEach(row => {
            if (!row._cellMap[column.column_id]) {
                row._cellMap[column.column_id] = {column_id: column.column_id, value: null};
            }
        });
        this.configureElement(this.element);
    }

    _onRemoveColumn(arg) {
        const c = this.configuration;
        const idx = typeof arg === 'number'
            ? arg
            : c.columns.findIndex(col => col.column_id === arg);
        if (idx < 0 || idx >= c.columns.length) return;

        const colId = c.columns[idx].column_id;
        c.columns.splice(idx, 1);
        c.rows.forEach(r => {
            delete r._cellMap[colId];
            delete r[colId];
        });
        this.configureElement(this.element);
    }

    assignListeners(element) {
        super.assignListeners(element);
        // no-op: individual cell widgets handle their own listeners
    }

    // --- Event dispatchers ------------------------------------------------
    _onSelectEvent(payload) {
        this.callbacks.get('event').call({
            id: this.id,
            event: "select",
            data: payload,
        });
    }

    _onInputEvent(payload) {
        this.callbacks.get('event').call({
            id: this.id,
            event: "input",
            data: payload,
        });
    }

    _onCheckboxEvent(payload) {
        this.callbacks.get('event').call({
            id: this.id,
            event: "checkbox",
            data: payload,
        });
    }

    _onButtonEvent(payload) {
        this.callbacks.get('event').call({
            id: this.id,
            event: "button",
            data: payload,
        });
    }

    validateInputCell({row, column, valid, value, message}) {
        const c = this.configuration;
        const rowIdx = c.rows.findIndex(r => r.row_id === row);
        const colIdx = c.columns.findIndex(col => col.column_id === column);
        if (rowIdx < 0 || colIdx < 0) {
            console.warn(`Invalid row/column for validation: ${row}, ${column}`);
            return;
        }

        const tr = this.tbody.children[rowIdx];
        const td = tr.children[colIdx];
        const input = td.querySelector('input.tableInput');
        const key = `${row}_${column}`;

        // always sync the value & our “prev” store
        input.value = value != null ? value : '';
        this._prevInputs[key] = input.value;

        if (valid) {
            input.classList.add('accepted');
            input.addEventListener('animationend',
                () => input.classList.remove('accepted'),
                {once: true}
            );
        } else {
            input.classList.add('error');
            input.addEventListener('animationend',
                () => input.classList.remove('error'),
                {once: true}
            );
            if (message) {
                this._showCellTooltip(input, message, true, true);
            }
        }
    }


    _showCellTooltip(inputEl, message, autoHide = false, isError = false) {
        if (!message) return;
        if (!this._tooltipEl) {
            this._tooltipEl = document.createElement('div');
            this._tooltipEl.classList.add('tiTooltipFloating');
            document.body.appendChild(this._tooltipEl);
        }
        this._tooltipEl.classList.toggle('tiTooltipFloatingError', isError);
        this._tooltipEl.textContent = message;

        // position it above the input
        const rect = inputEl.getBoundingClientRect();
        const tip = this._tooltipEl;

        // hide briefly so we can measure it
        tip.style.visibility = 'hidden';
        tip.style.opacity = '0';
        document.body.appendChild(tip);
        const tipRect = tip.getBoundingClientRect();

        const top = Math.max(8, rect.top - tipRect.height - 6) + window.scrollY;
        const left = rect.left + (rect.width - tipRect.width) / 2 + window.scrollX;

        Object.assign(tip.style, {
            top: `${top}px`,
            left: `${left}px`,
            visibility: 'visible',
            opacity: '1',
            transition: 'opacity 0.15s ease-in-out'
        });

        if (autoHide) {
            clearTimeout(this._tooltipTimeout);
            this._tooltipTimeout = setTimeout(() => this._hideCellTooltip(), 3000);
        }
    }

    _hideCellTooltip() {
        if (this._tooltipEl) {
            this._tooltipEl.style.visibility = 'hidden';
            this._tooltipEl.style.opacity = '0';
        }
    }

    initializeElement() {
    }

    resize() {
    }
}