export class BabylonObject {
    mesh;

    constructor(id, scene, config = {}) {
        this.id = id;
        this.scene = scene;
        this.config = config;
    }

    /**
     * @abstract
     */
    update(data) {
        throw new Error('Method not implemented.');
    }

    /**
     * @abstract
     */
    setPosition(position) {
        throw new Error('Method not implemented.');
    }

    /**
     * @abstract
     */
    setOrientation(orientation) {
        throw new Error('Method not implemented.');
    }

    /**
     * @abstract
     */
    highlight(state) {
        throw new Error('Method not implemented.');
    }

    /**
     * @abstract
     */
    onMessage(message) {
        throw new Error('Method not implemented.');
    }


    callFunction(function_name, args) {
        let fun = this[function_name];
        if (typeof fun === 'function') {
            fun.apply(this, args);
        }
    }

    setVisibility(visible) {
        this.mesh.isVisible = visible;
    }

    /**
     * @abstract
     * @param state
     */
    dim(state){
        throw new Error('Method not implemented.');
    }


}