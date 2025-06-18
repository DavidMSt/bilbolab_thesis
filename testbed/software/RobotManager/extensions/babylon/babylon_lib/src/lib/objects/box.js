import {BabylonObject} from "../objects.js";
import {CreateBox, StandardMaterial, Texture} from "@babylonjs/core";
import {coordinatesToBabylon, getBabylonColor, getBabylonColor3, loadTexture} from "../babylon_utils.js";
import {Quaternion} from '../quaternion.js'


// https://playground.babylonjs.com/#PCWRFE

export class BabylonBox extends BabylonObject {
    constructor(id, scene, config = {}) {
        super(id, scene, config);

        const default_config = {
            visible: true,
            color: [0.5, 0.5, 0.5],
            texture: '',
            texture_uscale: 1,
            texture_vscale: 1,
            wireframe: false,
            wireframe_width: 0.75,
            wireframe_color: [1, 0, 0, 1],
            alpha: 1,
            size: {
                x: 1,
                y: 1,
                z: 1,
            },
            position: [0, 0, 0],
            orientation: [1, 0, 0, 0],
            accept_shadows: false,
        }

        this.config = {...default_config, ...config};

        this.mesh = CreateBox('box', {
            height: this.config.size.z,
            width: this.config.size.x,
            depth: this.config.size.y
        }, this.scene);


        // # --- Material ----------------
        this.material = new StandardMaterial(this.scene);

        if (this.config.texture) {
            const tex = loadTexture(this.config.texture);
            this.material.diffuseTexture = new Texture(tex, this.scene);
            this.material.diffuseTexture.uScale = this.config.texture_uscale;
            this.material.diffuseTexture.vScale = this.config.texture_vscale;
            this.material.specularColor = getBabylonColor3([0, 0, 0]);
        } else {
            this.material.diffuseColor = getBabylonColor3(this.config.color);
        }

        this.mesh.material = this.material;
        this.mesh.material.alpha = this.config.alpha;

        if (this.config.wireframe) {
            this.mesh.enableEdgesRendering();
            this.mesh.edgesWidth = this.config.wireframe_width;
            this.mesh.edgesColor = getBabylonColor(this.config.wireframe_color);
        }

        this.setPosition(this.config.position);
        this.setOrientation(this.config.orientation);

        // --- SHADOW ---
        this.scene.shadowGenerator.addShadowCaster(this.mesh);

        this.mesh.acceptShadows = this.config.accept_shadows;

        // --- PICKING ---
        this.mesh.isPickable = true;
        this.mesh.metadata = {};
        this.mesh.metadata.object = this;
    }

    highlight(state) {
        return undefined;
    }

    onMessage(message) {
        return undefined;
    }

    setOrientation(orientation) {
        this.orientation = new Quaternion(orientation);
        this.mesh.rotationQuaternion = this.orientation.babylon();
    }

    setPosition(position) {
        let coords;

        if (Array.isArray(position)) {
            coords = position;
        } else if (typeof position === 'object' && position !== null &&
            'x' in position && 'y' in position && 'z' in position) {
            coords = [position.x, position.y, position.z];
        } else {
            throw new Error('Invalid position format. Expected [x, y, z] or {x, y, z}.');
        }

        this.position = coords;
        this.mesh.position = coordinatesToBabylon(coords);
    }


    update(data) {
        return undefined;
    }
}