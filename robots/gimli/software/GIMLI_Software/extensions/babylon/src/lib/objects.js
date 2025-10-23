import {Callbacks, splitPath} from "../../../gui/src/lib/helpers.js";

import {
    TransformNode,
    Quaternion,
    Vector3
} from "@babylonjs/core";
import {EventEmitter} from "events";
import {quatChanged, vecChanged} from "./babylon_utils";

// export class BabylonObject extends EventEmitter {
//
//     /** @type {string} */
//     id;
//
//     /** @type {BABYLON.Mesh} */
//     mesh = null;
//
//     /** @type {BABYLON.Scene} */
//     scene;
//
//     /** @type {Object} */
//     config;
//
//     /** @type {Object} */
//     data
//
//     /** @type {Babylon | BabylonObjectGroup} */
//     parent = null;
//
//     /** @type {boolean} */
//     visible = true;
//
//     /* === CONSTRUCTOR ============================================================================================== */
//     constructor(id, scene, payload = {}) {
//         super();
//         this.id = id;
//         this.scene = scene;
//         this.payload = payload;
//
//         this.config = this.payload.config;
//
//         this.data = this.payload.data || {};
//
//         // NEW: create a stable transform root for this object
//         this.root = new TransformNode(`${this.id}_root`, this.scene);
//         this.root.metadata = {object: this};
//
//
//         this.callbacks = new Callbacks()
//         this.callbacks.add('event');
//         this.callbacks.add('log');
//         this.callbacks.add('send_message');
//
//
//         // ---- tracking state for change detection (position/orientation) ----
//         /** @private */
//         this._rootObs = null;
//         /** @private */
//         this._rootLastFlag = null;
//         /** @private */
//         this._lastPose = {position: null, orientation: null};
//
//         this._addListeners();
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     /**
//      * @abstract
//      */
//     buildObject() {
//         throw new Error('Method not implemented.');
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     _getBabylonVisualization() {
//         if (this.parent) {
//             return this.parent.getBabylonVisualization();
//         }
//         return null;
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     static buildFromConfig(id, scene, payload) {
//         return new this(id, scene, payload);
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     /**
//      * @abstract
//      */
//     update(data) {
//         throw new Error('Method not implemented.');
//     }
//
//     // — basic transforms now apply to the root —
//     setPosition(position) {
//         this.position = Array.isArray(position)
//             ? (position.length === 2 ? [position[0], position[1], 0] : position)
//             : [position.x, position.y, position.z ?? 0];
//         // child classes decide world conversion; base just stores.
//     }
//
//     setOrientation(orientation) {
//         this.orientation = orientation instanceof Quaternion
//             ? orientation
//             : Quaternion.fromEulerAngles([typeof orientation === "number" ? orientation : 0, 0, 0], "zyx", true);
//         // base doesn’t write to root; subclasses decide when/how to convert.
//     }
//
//     setVisibility(visible) {
//         this.visible = visible;
//         if (this.root) this.root.setEnabled(!!visible);
//     }
//
//
//     delete() {
//         // remove observer before disposing
//         if (this._rootObs && this.root?.onAfterWorldMatrixUpdateObservable) {
//             this.root.onAfterWorldMatrixUpdateObservable.remove(this._rootObs);
//         }
//         this._rootObs = null;
//         this._rootLastFlag = null;
//
//         this.root?.dispose?.();
//         this.root = null;
//     }
//
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     /**
//      * @abstract
//      */
//     highlight(state) {
//         throw new Error('Method not implemented.');
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     /**
//      * @abstract
//      */
//     onMessage(message) {
//         throw new Error('Method not implemented.');
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     callFunction(function_name, args) {
//         let fun = this[function_name];
//         if (typeof fun === 'function') {
//             fun.apply(this, args);
//         }
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     /**
//      * @abstract
//      * @param state
//      */
//     dim(state) {
//         throw new Error('Method not implemented.');
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     _addListeners() {
//         if (!this.root?.onAfterWorldMatrixUpdateObservable) return;
//
//         // initialize lastFlag so the first callback only fires on the next *real* change
//         const wm = this.root.getWorldMatrix?.();
//         this._rootLastFlag = wm?.updateFlag ?? null;
//
//         // seed last known pose (so we can compare later)
//         this._lastPose = this._readAbsolutePose();
//
//         this._rootObs = this.root.onAfterWorldMatrixUpdateObservable.add(() => {
//             // 1) filter no-op updates using Babylon's updateFlag (matrix wasn't recomputed)
//             const m = this.root.getWorldMatrix?.();
//             const flag = m?.updateFlag ?? null;
//             if (this._rootLastFlag === flag) return;
//             this._rootLastFlag = flag;
//
//             // 2) read current absolute pose and compare with last
//             const cur = this._readAbsolutePose();
//             const moved =
//                 this._vecChanged(cur.position, this._lastPose?.position) ||
//                 this._quatChanged(cur.orientation, this._lastPose?.orientation);
//
//             if (!moved) return;
//
//             // 3) cache + notify
//             this._lastPose = cur;
//             this._onRootNodeUpdate();
//         });
//     }
//
//     /* -------------------------------------------------------------------------------------------------------------- */
//     _onRootNodeUpdate() {
//         this.emit('update');
//     }
//
// }


export class BabylonObject extends EventEmitter {

    /** @type {string} */
    id;

    /** @type {TransformNode} */
    root = null;

    /** @type {BABYLON.Mesh} */
    mesh = null;

    /** @type {BABYLON.Scene} */
    scene;

    /** @type {Object} */
    config;

    /** @type {Object} */
    data;

    /** @type {array} */
    position = []

    /** @type {Quaternion} */
    orientation = null;

    /** @type {Babylon | BabylonObjectGroup} */
    parent = null;

    /** @type {boolean} */
    visible = true;

    /** @private */
    _rootObs = null;
    /** @private */
    _rootLastFlag = null;
    /** @private */
    _lastPose = {position: null, orientation: null};

    /* === CONSTRUCTOR ============================================================================================== */
    constructor(id, babylon, payload = {}) {
        super();
        this.id = id;
        this.babylon = babylon;
        this.scene = babylon.scene;
        this.payload = payload;

        this.config = this.payload.config;
        this.data = this.payload.data || {};

        // stable transform root
        this.root = new TransformNode(`${this.id}_root`, this.scene);
        this.root.metadata = {object: this};

        this.callbacks = new Callbacks();
        this.callbacks.add("event");
        this.callbacks.add("log");
        this.callbacks.add("send_message");

        this._addListeners();
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * @abstract
     */
    buildObject() {
        throw new Error("Method not implemented.");
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    _getBabylonVisualization() {
        if (this.parent) {
            return this.parent.getBabylonVisualization();
        }
        return null;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    static buildFromConfig(id, scene, payload) {
        return new this(id, scene, payload);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * @abstract
     */
    update(data) {
        throw new Error("Method not implemented.");
    }

    // — basic transforms now apply to the root —
    setPosition(position) {
        this.position = Array.isArray(position)
            ? (position.length === 2 ? [position[0], position[1], 0] : position)
            : [position.x, position.y, position.z ?? 0];
    }

    setOrientation(orientation) {
        this.orientation = orientation instanceof Quaternion
            ? orientation
            : Quaternion.FromEulerAngles(
                typeof orientation === "number" ? orientation : 0,
                0,
                0
            );
    }

    setVisibility(visible) {
        this.visible = visible;
        if (this.root) this.root.setEnabled(!!visible);
    }

    delete() {
        // remove observer before disposing
        if (this._rootObs && this.root?.onAfterWorldMatrixUpdateObservable) {
            this.root.onAfterWorldMatrixUpdateObservable.remove(this._rootObs);
        }
        this._rootObs = null;
        this._rootLastFlag = null;

        this.root?.dispose?.();
        this.root = null;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * @abstract
     */
    highlight(state) {
        throw new Error("Method not implemented.");
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * @abstract
     */
    onMessage(message) {
        throw new Error("Method not implemented.");
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    callFunction(function_name, args) {
        let fun = this[function_name];
        if (typeof fun === "function") {
            fun.apply(this, args);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /**
     * @abstract
     * @param state
     */
    dim(state) {
        throw new Error("Method not implemented.");
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    _addListeners() {

        // World Matrix Update
        if (!this.root?.onAfterWorldMatrixUpdateObservable) return;

        // // initialize lastFlag so the first callback only fires on a real change
        // const wm = this.root.getWorldMatrix?.();
        // this._rootLastFlag = wm?.updateFlag ?? null;
        //
        // // seed last known pose
        // this._lastPose = this._readAbsolutePose();

        this._lastPose = {
            position: this.position,
            orientation: this.orientation,
        }

        this._rootObs = this.root.onAfterWorldMatrixUpdateObservable.add(() => {

            const poseChanged = vecChanged(this.position, this._lastPose.position) || quatChanged(this.orientation, this._lastPose.orientation);

            this._lastPose = {
                position: this.position,
                orientation: this.orientation,
            }
            if (poseChanged) {
                this._onRootNodeUpdate();
            }


        });
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    _onRootNodeUpdate() {
        this.emit("update");
        console.log("BabylonObject._onRootNodeUpdate");
    }

    /* ---------------- helpers: pose reading & comparisons ---------------- */

    /**
     * Read the root's absolute position & orientation.
     * @private
     */

    // _readAbsolutePose() {
    //     // position
    //     let p = null;
    //     if (this.root?.getAbsolutePosition) {
    //         const v = this.root.getAbsolutePosition();
    //         p = [v.x ?? 0, v.y ?? 0, v.z ?? 0];
    //     } else if (Array.isArray(this.position)) {
    //         p = [this.position[0] ?? 0, this.position[1] ?? 0, this.position[2] ?? 0];
    //     } else if (this.position && typeof this.position === "object") {
    //         p = [this.position.x ?? 0, this.position.y ?? 0, this.position.z ?? 0];
    //     } else {
    //         p = [0, 0, 0];
    //     }
    //
    //     // orientation (prefer world rotation via decomposition)
    //     let q = null;
    //     const wm = this.root?.getWorldMatrix?.();
    //     if (wm?.decompose) {
    //         const scale = new Vector3();
    //         const rot = new Quaternion();
    //         const trans = new Vector3();
    //         wm.decompose(scale, rot, trans);
    //         q = {x: rot.x, y: rot.y, z: rot.z, w: rot.w};
    //     } else if (this.root?.rotationQuaternion) {
    //         const r = this.root.rotationQuaternion;
    //         q = {x: r.x, y: r.y, z: r.z, w: r.w};
    //     } else if (this.orientation && typeof this.orientation === "object") {
    //         q = {
    //             x: this.orientation.x ?? 0,
    //             y: this.orientation.y ?? 0,
    //             z: this.orientation.z ?? 0,
    //             w: this.orientation.w ?? 1,
    //         };
    //     } else {
    //         q = {x: 0, y: 0, z: 0, w: 1};
    //     }
    //
    //     return {position: p, orientation: q};
    // }

    /** @private */

}

/* === BABYLON OBJECT GROUP ========================================================================================= */
export class BabylonObjectGroup {

    /** @type {string} */
    id;

    /** @type {BABYLON.Scene} */
    scene;

    /** @type {Babylon | BabylonObjectGroup} */
    parent = null;

    /** @type {Object} */
    config;

    /** @type {Object} */
    objects;

    /** @type {boolean} */
    visible = true;

    /* === CONSTRUCTOR ============================================================================================== */
    constructor(id, scene, config = {}, objects = {}) {

        this.id = id;
        this.scene = scene;
        this.config = config;
        this.objects = {};

        this.callbacks = new Callbacks()
        this.callbacks.add('event');
        this.callbacks.add('log');
        this.callbacks.add('send_message');

        // Build Objects from config

        if (Object.keys(objects).length > 0) {
            for (const [object_id, object_config] of Object.entries(objects)) {
                this.buildObjectFromConfig(object_id, object_config);
            }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    getBabylonVisualization() {
        if (this.parent) {
            return this.parent.getBabylonVisualization();
        }
        return null;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    addObject(object) {
        if (!(object instanceof BabylonObject) && !(object instanceof BabylonObjectGroup)) {
            throw new Error('Invalid object type. Expected BabylonObject or BabylonObjectGroup.');
        }
        if (object.id in this.objects) {
            throw new Error(`Object with ID ${object.id} already exists in this group.`);
        }
        this.objects[object.id] = object;
        object.parent = this;
        object.callbacks.get('event').register(this._onObjectEvent.bind(this));
        object.callbacks.get('log').register(this.callbacks.get('log').call.bind(this.callbacks.get('log')));
        object.callbacks.get('send_message').register(this.callbacks.get('send_message').call.bind(this.callbacks.get('send_message')));
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    removeObject(object) {
        if (!(object instanceof BabylonObject) && !(object instanceof BabylonObjectGroup)) {
            throw new Error('Invalid object type. Expected BabylonObject or BabylonObjectGroup.');
        }
        if (!(object.id in this.objects)) {
            throw new Error(`Object with ID ${object.id} does not exist in this group.`);
        }
        delete this.objects[object.id];
        object.parent = null;
        object.delete();
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    buildObjectFromConfig(object_payload) {
        const object_type = object_payload.type;
        const object_id = object_payload.id;
        const object_config = object_payload.config;

        const object_class = BABYLON_OBJECT_MAPPINGS[object_type];
        const object = object_class.buildFromConfig(object_config, this.scene);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    getObjectFromPath(path) {
        let firstSegment, remainder;

        [firstSegment, remainder] = splitPath(path);

        const childKey = `${this.id}/${firstSegment}`;

        const child = this.objects[childKey];

        if (!child) {
            console.warn(`Object with ID ${childKey} not found in group ${this.id}.`);
            return null;
        }

        if (!remainder) {
            return child;
        }

        if (child instanceof BabylonObjectGroup) {
            return child.getObjectFromPath(remainder);
        } else {
            console.warn(`Object with ID ${childKey} is not a group.`);
        }

        return null;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    setVisibility(visibility) {

        this.visible = visibility;
        // Go through all objects in the group and set visibility
        for (const object of Object.values(this.objects)) {
            object.setVisibility(visibility);
        }
    }

    /* === PRIVATE METHODS ========================================================================================== */
    _onObjectEvent(object, event) {
        this.callbacks.get('event').call(object, event);
    }


}