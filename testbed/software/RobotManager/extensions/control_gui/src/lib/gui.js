import {mountTerminal} from './terminal/terminal.js';
import {
    ButtonWidget,
    ClassicSliderWidget,
    DigitalNumberWidget,
    GUI_Object,
    InputWidget,
    MapWidget,
    MultiSelectWidget,
    MultiStateButtonWidget,
    ObjectGroup,
    PlotWidget,
    RotaryDialWidget,
    SliderWidget,
    StatusWidget,
    TableWidget,
    TextWidget
} from './objects.js'

import {splitPath, isObject, getColor} from './helpers.js';
import {Websocket} from './websocket.js';

const OBJECT_MAPPING = {
    'button': ButtonWidget,
    'slider': SliderWidget,
    'rotary_dial': RotaryDialWidget,
    'multi_state_button': MultiStateButtonWidget,
    'multi_select': MultiSelectWidget,
    'classic_slider': ClassicSliderWidget,
    'digital_number': DigitalNumberWidget,
    'text': TextWidget,
    'input': InputWidget,
    'status': StatusWidget,
    'table': TableWidget,
    'object_group': ObjectGroup,
    'map': MapWidget,
    'plot': PlotWidget,
}

const DEFAULT_BACKGROUND_COLOR = 'rgb(31,32,35)'
const DEBUG = true;

const GUI_WS_DEFAULT_PORT = 8100;

class Page {

    /** @type {Object} */
    objects = {};

    /** @type {Object} */
    callbacks = {};

    /** @type {Object} */
    configuration = {};

    /** @type {string} */
    id = '';


    /** @type {HTMLElement | null} */
    grid = null;


    /** @type {HTMLElement | null} */
    button = null;

    constructor(id, configuration = {}, objects = {}) {
        this.id = id;

        const default_configuration = {
            // rows: 16,
            // columns: 40,
            rows: 18,
            columns: 50,
            fillEmptyCells: true,
            color: 'rgba(40,40,40,0.7)',
            backgroundColor: DEFAULT_BACKGROUND_COLOR,
            text_color: 'rgba(255,255,255,0.7)',
            name: id,
        }

        this.configuration = {...default_configuration, ...configuration};

        this.parent = null;
        this.callbacks = {};
        this.objects = {};

        this.occupied_grid_cells = new Set();

        // Create the main grid container for this page that gets later swapped into the content container
        this.grid = document.createElement('div');
        this.grid.id = `page_${this.id}_grid`;
        this.grid.className = 'grid';

        // Make the number of rows and columns based on the configuration
        this.grid.style.gridTemplateRows = `repeat(${this.configuration.rows}, 1fr)`;
        this.grid.style.gridTemplateColumns = `repeat(${this.configuration.columns}, 1fr)`;

        this.grid.style.display = 'grid';

        // Fill the grid with empty cells
        if (this.configuration.fillEmptyCells) {
            this._fillContentGrid();
        }

        // Generate the button for this page that the category will later attach to the page bar
        this.button = this._generateButton();

        if (Object.keys(objects).length > 0) {
            this.buildObjectsFromDefinition(objects);
        }

    }

    /* ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯ */
    getObjectByPath(path) {
        // Example invocations:
        //   path = "button1"            → childKey = "/category1/page1/button1"
        //   path = "groupG/widgetX"     → childKey = "/category1/page1/groupG"
        //                                        then recurse with "widgetX"
        const [firstSegment, remainder] = splitPath(path);
        if (!firstSegment) {
            return null;
        }

        // Build the full‐UID key for the direct child:
        //   this.id is "/category1/page1"
        //   firstSegment might be "button1" or "groupG"
        const childKey = `${this.id}/${firstSegment}`;

        // Look up the widget or group in this.objects, which is keyed by full UID
        const child = this.objects[childKey];
        if (!child) {
            return null;
        }

        if (!remainder) {
            // No deeper path → return the widget or group itself
            return child;
        }

        // If there is more path to consume, the child must be an ObjectGroup
        if (child instanceof ObjectGroup) {
            return child.getObjectByPath(remainder);
        } else {
            // Cannot descend further if it’s not a group
            return null;
        }
    }

    /* ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯ */
    getGUI() {
        if (this.parent) {
            return this.parent.getGUI();
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    update(data) {
        console.log('Updating page:', this.id);
    }


    /* ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯ */
    handleAddMessage(data) {
        const object_config = data.config;
        if (object_config) {
            this.buildObjectFromConfig(object_config)
        }
    }

    /* ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯ */
    handleRemoveMessage(data) {

        const object_id = data.id;
        if (object_id) {
            const object = this.objects[object_id];
            if (object) {
                console.log(`Removing object ${object_id}`);
                delete this.objects[object_id];
                this.grid.removeChild(object.element);
                this._fillContentGrid();
            } else {
                console.warn(`Object ${object_id} not found`);
                console.log(this.objects);
            }
        }
    }

    /* ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯ */
    buildObjectsFromDefinition(objects) {
        for (const [id, config] of Object.entries(objects)) {
            this.buildObjectFromConfig(config);
        }
    }

    /* ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯ */
    buildObjectFromConfig(config) {
        const id = config.id;
        const type = config.type;
        const width = config.width;
        const height = config.height;
        const row = config.row;
        const col = config.column;
        const object_config = config.config;

        console.log(`Adding object ${id} of type ${type} at (${row},${col}) with config: ${object_config}`);

        // Check if the type is in the object mapping variable
        if (!OBJECT_MAPPING[type]) {
            console.warn(`Object type "${type}" is not defined.`);
            return;
        }

        const object_class = OBJECT_MAPPING[type];

        const object = new object_class(id, object_config);
        console.log(object);
        this.addObject(object, row, col, width, height);
    }

    /* ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯ */
    /**
     * Replace your old stub with this:
     * @param {GUI_Object} widget  — any widget subclass
     * @param {int} row
     * @param {int} col
     * @param {int} width
     * @param {int} height
     */
    addObject(widget, row, col, width, height) {
        if (!(widget instanceof GUI_Object)) {
            console.warn('Expected a GUI_Object, got:', widget);
            return;
        }

        if (!widget.id) {
            console.warn('Widget must have an ID');
            return;
        }

        if (this.objects[widget.id]) {
            console.warn(`Widget with ID "${widget.id}" already exists in the grid.`);
            return;
        }

        if (row < 0 || col < 0 || row >= this.configuration.rows || col >= this.configuration.columns) {
            console.warn(`Invalid grid coordinates: row=${row}, col=${col}`);
            return;
        }

        if (row + height - 1 > this.configuration.rows || col + width - 1 > this.configuration.columns) {
            console.warn(`Invalid grid dimensions: row=${row}, col=${col}, width=${width}, height=${height}`);
        }

        const newCells = this._getOccupiedCells(row, col, width, height);

        // Check for cell conflicts
        for (const cell of newCells) {
            if (this.occupied_grid_cells.has(cell)) {
                console.warn(`Grid cell ${cell} is already occupied. Cannot place widget "${widget.id}".`);
                return;
            }
        }

        // Mark the cells as occupied
        newCells.forEach(cell => this.occupied_grid_cells.add(cell));

        // Render the widget’s DOM and append into the main grid container
        const el = widget.render([row, col], [width, height]);
        this.grid.appendChild(el);
        this.objects[widget.id] = widget;

        widget.callbacks.event = this.onEvent.bind(this);

        if (this.configuration.fillEmptyCells) {
            this._fillContentGrid();
        }

    }

    /* -------------------------------------------------------------------------------------------------------------- */
    _generateButton() {
        let button = document.createElement('button');
        button.className = 'page_button';
        button.textContent = this.configuration.name;
        button.style.backgroundColor = this.configuration.color;
        button.style.color = getColor(this.configuration.text_color);
        button.classList.add('not-selected');

        return button;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    _getOccupiedCells(row, col, width, height) {
        const cells = [];
        for (let r = row; r < row + height; r++) {
            for (let c = col; c < col + width; c++) {
                cells.push(`${r},${c}`);
            }
        }
        return cells;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    // _fillContentGrid() {
    //     let occupied_cells = 0;
    //     this.grid
    //         .querySelectorAll('.placeholder')
    //         .forEach((el) => el.remove());
    //
    //     for (let row = 1; row < this.configuration.rows + 1; row++) {
    //         for (let col = 1; col < this.configuration.columns + 1; col++) {
    //             if (!this.occupied_grid_cells.has(`${row},${col}`)) {
    //                 const gridItem = document.createElement('div');
    //                 gridItem.className = 'placeholder';
    //                 // gridItem.textContent = `${row},${col}`;
    //                 gridItem.style.fontSize = '6px';
    //                 gridItem.style.color = 'rgba(255,255,255,0.5)';
    //                 this.grid.appendChild(gridItem);
    //             } else {
    //                 occupied_cells++;
    //             }
    //         }
    //     }
    //
    //     console.log(`Page "${this.id}" has ${occupied_cells} occupied cells.`);
    // }

    /* -------------------------------------------------------------------------------------------------------------- */
    _fillContentGrid() {
        let occupied_cells = 0;

        // Remove any existing placeholders
        this.grid
            .querySelectorAll('.placeholder')
            .forEach((el) => el.remove());

        for (let row = 1; row < this.configuration.rows + 1; row++) {
            for (let col = 1; col < this.configuration.columns + 1; col++) {
                if (!this.occupied_grid_cells.has(`${row},${col}`)) {
                    const gridItem = document.createElement('div');
                    gridItem.className = 'placeholder';

                    // Set a tooltip showing the 1-based row and column
                    gridItem.title = `Row ${row}, Column ${col}`;

                    gridItem.style.fontSize = '6px';
                    gridItem.style.color = 'rgba(255,255,255,0.5)';
                    this.grid.appendChild(gridItem);
                } else {
                    occupied_cells++;
                }
            }
        }

        // console.log(`Page "${this.id}" has ${occupied_cells} occupied cells.`);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    onEvent(event) {
        // Check if there is an 'event' callback for this page
        if (DEBUG) {
            console.log(`[Page ID: ${this.id}] Event received:`, event);
        }
        if (this.callbacks.event) {
            this.callbacks.event(event);
        }
    }
}

/* ================================================================================================================== */
class Category {

    /** @type {Object<string,Page>} */
    pages = {};

    /** @type {Page|null} */
    page = null;

    /** @type {Object<string,Category>} */
    categories = {};

    /** @type {Object} */
    callbacks = {};

    /** @type {Object} */
    configuration = {};

    /** @type {string} */
    id = '';

    /** @type {HTMLElement|null} */
    button = null;

    /** @type {Object<number,HTMLElement|null>} */
    page_buttons = {};

    /** @type {HTMLElement|null} */
    page_grid = null;

    /** @type {HTMLElement|null} */
    content_grid = null;


    /**
     * @param {string} id
     * @param {Object} [configuration={}]
     * @param {Object} [pages={}]         – map of page-definitions
     * @param {Object} [categories={}]    – map of subcategory-definitions
     */
    constructor(id, configuration = {}, pages = {}, categories = {}) {
        this.id = id;

        const default_configuration = {
            name: id,
            collapsed: false,
            color: 'rgba(40,40,40,0.7)',
            text_color: 'rgba(255,255,255,0.7)',
            icon: null,
            top_icon: null,
            number_of_pages: +getComputedStyle(document.documentElement).getPropertyValue('--page_bar-cols'),
            max_pages: +getComputedStyle(document.documentElement).getPropertyValue('--page_bar-cols'),
        };

        this.configuration = {...default_configuration, ...configuration};
        this.parent = null;

        this.callbacks = {};
        this.pages = {};
        this.categories = {};


        this.page = null;

        // main button for this category
        this.button = this._generateButton();

        // slots for page-buttons
        this.page_buttons = {};
        for (let i = 0; i < this.configuration.number_of_pages; i++) {
            this.page_buttons[i] = null;
        }

        // container for page buttons
        this._createPageGrid();

        // build out any initially defined pages & categories
        if (Object.keys(pages).length > 0) {
            this.buildPagesFromDefinition(pages);
        }
        if (Object.keys(categories).length > 0) {
            this.buildCategoriesFromDefinition(categories);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Look up something by path, descending into sub-categories first, then pages.
     * @param {string} path
     * @returns {Category|Page|null}
     */
    getObjectByPath(path) {
        // console.log(`[Category ID: ${this.id}] getObjectByPath called with:`, path);
        // console.log(this)
        const [firstSegment, remainder] = splitPath(path);
        if (!firstSegment) return null;

        const fullKey = `${this.id}/${firstSegment}`;
        // 1) Sub‐category?
        const subCat = this.categories[fullKey];
        if (subCat) {
            if (!remainder) return subCat;
            return subCat.getObjectByPath(remainder);
        }

        // 2) Page?
        const page = this.pages[fullKey];
        if (!page) return null;
        if (!remainder) return page;
        return page.getObjectByPath(remainder);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    getGUI() {
        if (this.parent instanceof Category) {
            return this.parent.getGUI();
        } else if (this.parent instanceof GUI) {
            return this.parent;
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    update(data) {
        console.log('Updating category:', this.id);
        console.warn('Category update is not yet implemented.');
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    handleAddMessage(data) {

        const object_type = data.type

        console.log(`Category: Handling add message of ${object_type} type`);

        switch (object_type) {
            case 'page':
                this.buildPageFromDefinition(data.config);
                break;
            case 'category':
                this.buildCategoryFromDefinition(data.config);

                const gui = this.getGUI();
                if (gui) {
                    gui.renderCategoryTree();
                }
                break;
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    handleRemoveMessage(data) {
        const object_type = data.type

        console.log(`Category: Handling remove message of ${object_type} type`);
        switch (object_type) {
            case 'page':
                const page_id = data.id
                const page = this.pages[page_id]
                if (page) {

                    // Remove the page's button
                    page.button.remove()
                    page.grid.remove()
                    delete this.pages[page_id]

                    // Switch active page
                    if (this.page === page) {
                        // check the length of the this.pages array. If bigger than 0, then choose the first one
                        if (Object.keys(this.pages).length > 0) {
                            this.setPage(Object.keys(this.pages)[0])
                        } else {
                            // if the length is 0, then set the page to null
                            this.setPage(null)
                        }
                    }

                }
                break;
            case 'category':
                const category_id = data.id;
                const category = this.categories[category_id];
                if (category) {
                    // category.content_grid.remove()
                    delete this.categories[category_id];
                    // Switch active category if it was this category

                    if (isObject(category.id, this.getGUI().category.id)) {
                        this.getGUI().setCategory(this.id);
                    }


                    this.getGUI().renderCategoryTree();
                }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Build multiple pages from a definition map
     * @param {Object<string,*>} pages
     */
    buildPagesFromDefinition(pages) {
        for (const [_, config] of Object.entries(pages)) {
            this.buildPageFromDefinition(config);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Build a single page from its definition and add it
     * @param {{id:string, config:Object, objects:Object, position?:number}} page_definition
     */
    buildPageFromDefinition(page_definition) {
        const new_page = new Page(
            page_definition.id,
            page_definition.config,
            page_definition.objects
        );
        this.addPage(new_page, page_definition.position);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Build multiple subcategories from a definition map
     * @param {Object<string,*>} categories
     */
    buildCategoriesFromDefinition(categories) {
        console.log('Building categories:', categories);
        for (const [_, config] of Object.entries(categories)) {
            this.buildCategoryFromDefinition(config);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Build a single subcategory from its definition and add it
     * @param {{id:string, config:Object, pages:Object, categories:Object, position?:number}} cat_definition
     */
    buildCategoryFromDefinition(cat_definition) {
        const new_category = new Category(
            cat_definition.id,
            cat_definition.config,
            cat_definition.pages || {},
            cat_definition.categories || {}
        );
        this.addCategory(new_category, cat_definition.position);
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    _generateButton() {
        const {name, icon, top_icon, text_color} = this.configuration;

        // 1) create the <button>
        const button = document.createElement('button');
        button.classList.add('category-button', 'not-selected');
        button.style.color = text_color;

        // 2) left-icon slot (fixed size, may remain empty)
        const iconSlot = document.createElement('span');
        iconSlot.className = 'category-button__icon';
        if (this.configuration.icon) {
            if (typeof this.configuration.icon === 'string' && this.configuration.icon.match(/\.(png|jpg|jpeg|svg)$/i)) {
                const img = document.createElement('img');
                img.src = this.configuration.icon;
                img.alt = '';
                iconSlot.appendChild(img);
            } else {
                iconSlot.textContent = this.configuration.icon; // assume emoji or text
            }
        }
        button.appendChild(iconSlot);

        // 3) text label
        const textSpan = document.createElement('span');
        textSpan.className = 'category-button__text';
        textSpan.textContent = name;
        button.appendChild(textSpan);

        // 4) optional top‐icon
        if (top_icon) {
            const topSlot = document.createElement('span');
            topSlot.className = 'category-button__top-icon';
            topSlot.textContent = top_icon;
            button.appendChild(topSlot);
        }

        return button;
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    _createPageGrid() {
        this.page_grid = document.createElement('div');
        this.page_grid.id = `page_${this.id}_grid`;
        this.page_grid.className = 'page_bar_grid';
    }

    hidePages() {
        Object.values(this.pages).forEach(pg => {
            pg.grid.style.display = 'none';
        });
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Add a ControlGUI_Page to this category
     * @param {Page} page
     * @param {number|null} position
     */
    addPage(page, position = null) {
        if (this.pages[page.id]) {
            console.warn(`Page with ID "${page.id}" already exists in category "${this.id}".`);
            return;
        }

        // find or validate slot
        if (position !== null) {
            if (this.page_buttons[position - 1] !== null) {
                console.warn(`Position ${position} already used in category "${this.id}".`);
                return;
            }
        } else {
            for (let i = 1; i <= this.configuration.number_of_pages; i++) {
                if (this.page_buttons[i - 1] === null) {
                    position = i;
                    break;
                }
            }
            if (position === null) {
                console.warn(`No free page slots in category "${this.id}".`);
                return;
            }
        }

        // wire up button
        this.page_buttons[position - 1] = page.button;
        page.button.style.gridRow = '1';
        page.button.style.gridColumn = String(position);
        page.button.addEventListener('click', () => this.setPage(page.id));
        this.page_grid.appendChild(page.button);

        // register
        this.pages[page.id] = page;
        page.parent = this;
        page.callbacks.event = this.onEvent.bind(this);

        // Add the pages grid to my content grid
        if (this.content_grid) {
            // only attach it if it isn’t already in the DOM
            if (page.grid.parentNode !== this.content_grid) {
                this.content_grid.appendChild(page.grid);
            }
            Object.assign(page.grid.style, {
                position: 'absolute',
                top: '0',
                left: '0',
                width: '100%',
                height: '100%',
                display: 'none',
            });
        }

        // if first page, show it
        if (this.page === null) {
            this.setPage(page.id);
        }
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Add a ControlGUI_Category as a nested subcategory
     * @param {Category} category
     * @param {number|null} position   – currently unused for UI
     */
    addCategory(category, position = null) {
        if (this.categories[category.id]) {
            console.warn(`Category with ID "${category.id}" already exists under "${this.id}".`);
            return;
        }
        this.categories[category.id] = category;
        category.parent = this;
        category.callbacks.event = this.onEvent.bind(this);
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Render page-buttons into `container` and absolutely‐position all page.grids
     * (unchanged from before)
     */
    buildCategory(container, content_grid) {

        const gui = this.getGUI();
        // 1) collapse or show the entire page‐bar row
        if (gui) {
            gui.showPageBar(this.configuration.max_pages > 1);
        }

        // 2) populate (or clear) the page‐bar itself
        container.innerHTML = '';
        if (this.configuration.max_pages > 1) {
            container.style.display = '';
            container.appendChild(this.page_grid);
        } else {
            // we’ve already hidden the <nav>, but just in case:
            container.style.display = 'none';
        }

        this.content_grid = content_grid;
        this.content_grid.style.position = 'relative';

        Object.values(this.pages).forEach(page => {
            if (page.grid.parentNode !== this.content_grid) {
                this.content_grid.appendChild(page.grid);
                Object.assign(page.grid.style, {
                    position: 'absolute',
                    top: '0',
                    left: '0',
                    width: '100%',
                    height: '100%',
                    display: 'none',
                });
            } else {
                page.grid.style.display = 'none';
            }
        });

        const startId = this.page ? this.page.id : Object.keys(this.pages)[0];
        if (startId) this.setPage(startId);
        else this._renderEmpty(container, content_grid);
    }

    _renderEmpty(container, content_grid) {
        content_grid.innerHTML = '';
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * Switch visible page (unchanged)
     * @param {string|Page} pageOrId
     */
    setPage(pageOrId) {
        const id = pageOrId instanceof Page ? pageOrId.id : pageOrId;
        const page = this.pages[id];
        if (!page) {
            console.warn(`Page "${id}" not found in category "${this.id}".`);
            return;
        }

        Object.values(this.pages).forEach(p => {
            p.grid.style.display = 'none';
            p.button.classList.remove('selected');
        });

        page.grid.style.display = 'grid';
        page.button.classList.add('selected');
        window.dispatchEvent(new Event('resize'));
        this.page = page;
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    onEvent(event) {
        console.log(`[Category ID: ${this.id}] Event received:`, event);
        if (this.callbacks.event) {
            this.callbacks.event(event);
        }
    }
}


/* ================================================================================================================== */
export class GUI {

    grid = null;
    content = null;
    head_bar = null;
    head_bar_grid = null;
    page_bar = null;
    category_bar = null;
    terminal_container = null;
    rows = 0;
    cols = 0;

    /** @type {Object} */
    category_buttons = {};

    /** @type {Object} */
    configuration = {};

    /** @type {boolean} */
    connected = false;

    /* ===============================================================================================================*/
    constructor(rootContainer, configuration = {}) {

        const default_configuration = {
            number_of_categories: 10,
            show_category_bar: true
            // auto_hide_category_bar: true,
        }

        this.rootContainer = rootContainer;
        this.configuration = {...default_configuration, ...configuration};
        this.drawGUI();
        this.showCategoryBar(this.configuration.show_category_bar);

        this.drawStatusBarGrid()

        this._initializeTerminal();

        this.category = null;
        this.categories = {}
        this.category_buttons = {}

        for (let i = 0; i < this.configuration.number_of_categories; i++) {
            this.category_buttons[i] = null;
        }

        this.addLogo();
        this.addConnectionIndicator();

        const websocket_host = import.meta.env.VITE_WS_HOST || window.location.hostname;
        const websocket_port = parseInt(import.meta.env.VITE_WS_PORT, 10) || GUI_WS_DEFAULT_PORT;

        console.log(`Connecting to websocket at ${websocket_host}:${websocket_port}`);

        this.websocket = new Websocket({host: websocket_host, port: websocket_port})
        this.websocket.connect();
        this.websocket.on('message', this.onWsMessage.bind(this));
        this.websocket.on('connected', this.onWsConnected.bind(this));
        this.websocket.on('close', this.onWSDisconnected.bind(this));
        this.websocket.on('error', this.onWsError.bind(this));

        this.resetGUI();
    }

    /* ===============================================================================================================*/
    drawGUI() {
        // clear any existing content
        this.rootContainer.innerHTML = '';

        // ── HEADER / HEADBAR ────────────────────────────────────────────────────
        this.head_bar = document.createElement('header');
        this.head_bar.id = 'headbar';
        this.head_bar_grid = document.createElement('div');
        this.head_bar_grid.id = 'headbar_grid';
        this.head_bar.appendChild(this.head_bar_grid);
        this.rootContainer.appendChild(this.head_bar);

        // ── SIDE PLACEHOLDER ───────────────────────────────────────────────────
        this.side_placeholder = document.createElement('div');
        this.side_placeholder.id = 'side_placeholder';
        this.rootContainer.appendChild(this.side_placeholder);

        // ── ROBOT STATUS BAR ───────────────────────────────────────────────────
        this.robot_status_bar = document.createElement('nav');
        this.robot_status_bar.id = 'robot_status_bar';
        this.robot_status_bar_grid = document.createElement('div');
        this.robot_status_bar_grid.id = 'robot_status_bar_grid';
        this.robot_status_bar.appendChild(this.robot_status_bar_grid);
        this.rootContainer.appendChild(this.robot_status_bar);

        // ── PAGE BAR ───────────────────────────────────────────────────────────
        this.page_bar = document.createElement('nav');
        this.page_bar.id = 'page_bar';
        this.page_bar_grid = document.createElement('div');
        this.page_bar_grid.id = 'page_bar_grid';
        this.page_bar_grid.className = 'page_bar_grid';
        this.page_bar.appendChild(this.page_bar_grid);
        this.rootContainer.appendChild(this.page_bar);

        // ── CATEGORY BAR ───────────────────────────────────────────────────────
        this.category_bar = document.createElement('aside');
        this.category_bar.id = 'category_bar';

        // <— instead of a <div class="grid">, make a <ul> for nesting
        this.category_bar_list = document.createElement('ul');
        this.category_bar_list.id = 'category_bar_list';
        this.category_bar.appendChild(this.category_bar_list);

        this.rootContainer.appendChild(this.category_bar);


        // ── MAIN CONTENT ───────────────────────────────────────────────────────
        this.content = document.createElement('main');
        this.content.id = 'content';
        this.rootContainer.appendChild(this.content);

        // ── FOOTER / TERMINAL ──────────────────────────────────────────────────
        const footer = document.createElement('footer');
        footer.id = 'bottombar';

        this.terminal_container = document.createElement('div');
        this.terminal_container.id = 'terminal-container';
        this.terminal_container.className = 'bottombar__left';

        const bottombarRight = document.createElement('div');
        bottombarRight.className = 'bottombar__right';

        footer.appendChild(this.terminal_container);
        footer.appendChild(bottombarRight);
        this.rootContainer.appendChild(footer);
    }

    /**
     * Show or hide the page‐bar row and
     * collapse/restore the third grid‐row on #app.
     * @param {boolean} show
     */
    showPageBar(show) {
        if (show) {
            // put the page_bar back…
            this.page_bar.style.display = '';
            // …and restore your default CSS template‐rows
            this.rootContainer.style.gridTemplateRows = '';
        } else {
            // hide the page_bar completely
            this.page_bar.style.display = 'none';
            // collapse that row to zero and let content fill it
            this.rootContainer.style.gridTemplateRows =
                'var(--headbar-height) ' +
                'var(--robot-status-bar-height) ' +
                '0 ' +               // ← collapse the “pages” row
                '1fr ' +             // ← content now starts here
                'var(--bottom-height)';
        }
    }

    /* ===============================================================================================================*/
    /**
     * Toggle category-bar on/off.
     * When off, collapses the first grid column to zero,
     * so page-bar, content and bottombar all fill the remaining space.
     */
    showCategoryBar(show) {
        // Do we currently have the bar hidden?

        if (show) {
            // → SHOW IT AGAIN
            this.category_bar.style.display = '';
            // restore the grid-template-columns from your CSS
            this.rootContainer.style.gridTemplateColumns = '';
        } else {
            // → HIDE IT
            this.category_bar.style.display = 'none';
            // collapse the first column, let the 2nd column fill 100%
            this.rootContainer.style.gridTemplateColumns = '0 1fr';
        }
    }


    /* ===============================================================================================================*/
    getObjectByUID(uid) {

        const trimmed = uid.replace(/^\/+|\/+$/g, '');

        const [gui_segment, category_remainder] = splitPath(trimmed);

        if (!gui_segment || gui_segment !== this.id) {
            console.warn(`UID "${uid}" does not match this GUI's ID "${this.id}".`);
            return null;
        }
        if (!category_remainder) {
            return this;
        }

        const [categorySegment, remainder] = splitPath(category_remainder);
        const fullKey = `${this.id}/${categorySegment}`;
        // 1) Sub‐category?
        const subCat = this.categories[fullKey];
        if (subCat) {
            if (!remainder) return subCat;
            return subCat.getObjectByPath(remainder);
        } else {
            console.warn(`Category "${categorySegment}" not found in GUI.`);
        }
    }

    /* ===============================================================================================================*/
    resetGUI() {
        // Empty the content
        this.content.innerHTML = '';
        this.category_bar_list.innerHTML = '';
        this.page_bar.innerHTML = '';

        // Delete all categories that are currently stored
        this.categories = {};
        this.category = null;

        for (let i = 0; i < this.configuration.number_of_categories; i++) {
            this.category_buttons[i] = null;
        }

        // Add the placeholder in the middle of the content area
        const placeholder = document.createElement('div');
        placeholder.className = 'content_placeholder';
        placeholder.innerHTML = `
            <span class="placeholder_title">Not connected</span>
            <span class="placeholder_info">${this.websocket.url}</span>
            `;
        this.content.appendChild(placeholder);

        this.msgRateDisplay.textContent = '-----';
    }

    /* ===============================================================================================================*/
    addLogo() {

        const logoLink = document.createElement('a')
        logoLink.href = 'https://github.com/dustin-lehmann/bilbolab' // Change to your desired URL
        logoLink.className = 'logo_link'
        logoLink.target = '_blank' // Opens in a new tab
        logoLink.rel = 'noopener noreferrer' // Security best practice

        const logo = document.createElement('img')
        logo.src = new URL('./symbols/bilbolab_logo.png', import.meta.url).href
        logo.alt = 'Logo'
        logo.className = 'bilbolab_logo'

        logoLink.appendChild(logo)
        this.head_bar_grid.appendChild(logoLink)

    }

    /* ===============================================================================================================*/
    addConnectionIndicator() {

        // ——— websocket status & rate indicator ———
        this.msgTimestamps = [];
        this.msgRateWindow = 1;
        this.blinkThrottle = 100;      // ms between blinks
        this._lastBlinkTime = 0;

        // create a container in the head_bar_grid
        const statusContainer = document.createElement('div');
        statusContainer.style.gridRow = '1 / span 2';                       // top row
        statusContainer.style.gridColumn = `${String(this.headbar_cols - 1)} / span 2`; // far right
        statusContainer.style.justifySelf = 'end';
        statusContainer.style.marginRight = '10px';
        statusContainer.style.paddingRight = '10px';
        statusContainer.style.display = 'flex';
        statusContainer.style.alignItems = 'center';
        statusContainer.style.gap = '8px';


        // the little circle
        this.statusIndicator = document.createElement('div');
        this.statusIndicator.className = 'status-indicator';

        // the “X M/s” text
        this.msgRateDisplay = document.createElement('span');
        this.msgRateDisplay.className = 'msg-rate';
        this.msgRateDisplay.textContent = '-----';

        statusContainer.appendChild(this.statusIndicator);
        statusContainer.appendChild(this.msgRateDisplay);
        this.head_bar_grid.appendChild(statusContainer);
    }

    /**
     * Register a new top‐level category and rebuild the sidebar.
     *
     * @param {Category} category
     * @param {number|null} position  — no longer used; kept for backward compatibility
     */
    addCategory(category, position = null) {
        // 1) Dedupe
        category.parent = this;
        if (this.categories[category.id]) {
            console.warn(`Category "${category.id}" already exists.`);
            return;
        }

        // 2) Register it
        this.categories[category.id] = category;
        category.callbacks.event = this._onEvent.bind(this);

        // 3) If this is the very first category, select it immediately
        if (this.category === null) {
            this.setCategory(category.id);
        }


        // 4) Rebuild the nested sidebar list so it shows the new category
        this.renderCategoryTree();
    }

    /* ===============================================================================================================*/

    /* ===============================================================================================================*/
    setCategory(category_id) {

        // Try to retrieve the category from the object tree
        const category = this.getObjectByUID(category_id);

        if (!category) {
            console.warn(`Category "${category_id}" not found.`);
            return;
        }
        console.log('Setting category to: ', category_id);

        // 1) Hide the page from the active category
        // 2) Make the category button unselected
        if (this.category) {
            this.category.hidePages();
            this.category.button.classList.remove('selected');
        }

        // 3) Save the category as the new active category
        this.category = category;

        this.category.buildCategory(this.page_bar, this.content);

        this.renderCategoryTree();
    }

    // ─── Replace this entire method in your GUI class ───────────────────────
    renderCategoryTree() {
        const container = this.category_bar_list;
        container.innerHTML = '';

        // Grab indent size once
        const indentPx = parseInt(
            getComputedStyle(document.documentElement)
                .getPropertyValue('--category-indent-step'),
            10
        );

        const build = (cats, level = 0) => {
            cats.forEach(cat => {
                // 1) make the <li>
                const li = document.createElement('li');
                li.className = 'category-item';
                li.style.position = 'relative';
                li.style.paddingLeft = `${level * indentPx}px`;

                // 2) insert one <span> per ancestor-level to draw its vertical line
                for (let lv = 1; lv <= level; lv++) {
                    const line = document.createElement('span');
                    line.className = 'connector-line';
                    // position each at its own indent offset
                    line.style.left = `${lv * indentPx - 5}px`;
                    li.appendChild(line);
                }

                // 3) double-click toggles open/closed if it has children
                if (Object.keys(cat.categories).length > 0) {

                    cat.button.classList.add('has-children');

                    li.addEventListener('dblclick', e => {
                        e.stopPropagation();
                        cat.configuration.collapsed = !cat.configuration.collapsed;
                        this.renderCategoryTree();
                    });

                    cat.button.classList.toggle('collapsed', cat.configuration.collapsed);

                } else {
                    cat.button.classList.remove('has-children');
                    cat.button.classList.remove('collapsed');
                }

                // 4) style/select button
                cat.button.classList.toggle('selected',
                    this.category && this.category.id === cat.id
                );
                cat.button.classList.toggle('not-selected',
                    !(this.category && this.category.id === cat.id)
                );

                // 5) click selects
                li.appendChild(cat.button);
                li.addEventListener('click', () => this.setCategory(cat.id));

                container.appendChild(li);

                // 6) recurse into open subcategories
                if (!cat.configuration.collapsed) {
                    build(Object.values(cat.categories), level + 1);
                }
            });
        };

        build(Object.values(this.categories), 0);
    }


    /* ===============================================================================================================*/
    _initializeTerminal() {
        mountTerminal('#' + this.terminal_container.id);
    }

    /* ===============================================================================================================*/
    drawStatusBarGrid() {
        for (let row = 0; row < this.robot_status_bar_rows; row++) {
            for (let col = 0; col < this.robot_status_bar_cols; col++) {
                const gridItem = document.createElement('div');
                gridItem.className = 'robot_status_bar_cell';
                // gridItem.textContent = `${row},${col}`;  // Optional: for debugging
                this.robot_status_bar_grid.appendChild(gridItem);
            }
        }
    }


    /* ===============================================================================================================*/
    drawHeadBarGrid() {
        for (let row = 0; row < this.headbar_rows; row++) {
            for (let col = 0; col < this.headbar_cols; col++) {
                const gridItem = document.createElement('div');
                gridItem.className = 'headbar_cell';
                // gridItem.textContent = `${row},${col}`;  // Optional: for debugging
                this.head_bar_grid.appendChild(gridItem);
            }
        }
    }


    /* ===============================================================================================================*/
    onWsConnected() {
        this.connected = true;
        this.setConnectionStatus(true);

        const handshake_message = {
            type: 'handshake',
            data: {
                'client_type': 'frontend'
            }
        }
        this.websocket.send(handshake_message);
    }

    onWSDisconnected() {
        this.connected = false;
        this.setConnectionStatus(false);
        this.resetGUI();
    }

    onWsMessage(msg) {
        switch (msg.type) {
            case 'init':
                this._initialize(msg);
                break;
            case 'update':
                this._update(msg);
                break;
            case 'add':
                this._handleAddMessage(msg);
                break;
            case 'remove':
                this._handleRemoveMessage(msg);
                break;
            case 'widget_message':
                this._handleMessageForWidget(msg);
                break;
            default:
                console.warn('Unknown message type', msg.type);
        }

        this._recordMessage();
    }

    onWsError(err) {

    }

    /* ===============================================================================================================*/
    _onEvent(event) {
        console.log('GUI Event received: ', event)
        const message = {
            'type': 'event', 'id': event.id, 'data': event,
        }
        if (this.connected) {
            this.websocket.send(message);
        }
    }

    /* ===============================================================================================================*/
    _initialize(msg) {
        console.log('Initializing control GUI');

        // Check if msg has a field name configuration, if yes extract it
        if (msg.configuration) {
            const config = msg.configuration;
            console.log('Configuration received: ', config);

            this.id = config.id || 'gui';
            // TODO: Here we have to set some properties, such as show category bar or auto_hide

            if (config.categories) {
                for (let id in config.categories) {
                    console.log("Adding category: ", config.categories[id].id);
                    console.log("Configuration:", config.categories[id].config)
                    console.log(config.categories[id])
                    const category = new Category(config.categories[id].id,
                        config.categories[id].config,
                        config.categories[id].pages,
                        config.categories[id].categories,);

                    this.addCategory(category);
                }
            }

        }
    }

    /* ===============================================================================================================*/
    _update(msg) {
        const object = this.getObjectByUID(msg.id);
        if (!object) {
            console.warn(`Object with UID "${msg.id}" not found.`);
            return;
        }
        object.update(msg.data);
    }

    /* ===============================================================================================================*/
    _handleMessageForWidget(message) {
        const object = this.getObjectByUID(message.id);
        if (object) {
            object.onMessage(message.data);
        }
    }

    /* ===============================================================================================================*/
    _handleAddMessage(message) {
        const data = message.data;
        // Get the object we want to add something to
        const parent = this.getObjectByUID(data.parent);

        if (!parent) {
            console.warn(`Received add message for unknown parent "${data.parent}"`);
            return
        }
        parent.handleAddMessage(data);
    }

    /* ===============================================================================================================*/
    _handleRemoveMessage(message) {
        console.log('Received remove message: ', message);
        const data = message.data;
        const parent = this.getObjectByUID(data.parent);
        if (!parent) {
            console.warn(`Received remove message for unknown parent "${data.parent}"`);
            return;
        }
        parent.handleRemoveMessage(data);
    }

    /* ===============================================================================================================*/
    /**
     * Call on WebSocket open/close
     */
    setConnectionStatus(connected) {
        if (connected) {
            this.statusIndicator.classList.add('connected');
            const placeholder = this.content.querySelector('.content_placeholder');
            if (placeholder) placeholder.remove();
        } else {
            this.statusIndicator.classList.remove('connected');
            this.msgRateDisplay.textContent = '---';
        }
    }

    /* ===============================================================================================================*/
    /**
     * Call this for every incoming message event
     */
    _recordMessage() {
        const now = Date.now();
        this.msgTimestamps.push(now);
        // drop anything older than the window
        const cutoff = now - this.msgRateWindow * 1000;
        while (this.msgTimestamps.length && this.msgTimestamps[0] < cutoff) {
            this.msgTimestamps.shift();
        }
        this._updateMessageRate();
        this._maybeBlink();
    }

    /* ===============================================================================================================*/
    /**
     * Recompute and display the messages/sec
     */
    _updateMessageRate() {
        const count = this.msgTimestamps.length;
        const rate = count / this.msgRateWindow;
        this.msgRateDisplay.textContent = rate.toFixed(1) + ' M/s';
    }

    /* ===============================================================================================================*/
    /**
     * Blink the status indicator at most once per blinkThrottle ms
     */
    _maybeBlink() {
        const now = Date.now();
        if (now - this._lastBlinkTime < this.blinkThrottle) return;
        this._lastBlinkTime = now;
        this.statusIndicator.classList.add('blink');
        this.statusIndicator.addEventListener(
            'animationend',
            () => this.statusIndicator.classList.remove('blink'),
            {once: true}
        );
    }

}

