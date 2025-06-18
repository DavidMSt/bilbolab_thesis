import {BabylonObject} from "../objects.js";
import {
    StandardMaterial,
    SceneLoader,
    DynamicTexture,
    TransformNode,
    Vector3,
    MeshBuilder
} from "@babylonjs/core";
import {
    coordinatesToBabylon,
    getBabylonColor3,
    loadModel,
    getHTMLColor,
} from "../babylon_utils.js";
import {Quaternion} from '../quaternion.js';

export class BabylonFrodo extends BabylonObject {
    constructor(id, scene, config = {}) {
        super(id, scene, config);

        const default_config = {
            model: 'frodo_generic.babylon',
            text: '',
            text_color: [1, 1, 1],
            color: [1, 0, 0],
            scaling: 1,
            position: [0, 0, 0],
            orientation: [1, 0, 0, 0],
            z_offset: 0,
            fov_radius: 1,
            fov_angle_deg: 120,
        };

        this.config = {...default_config, ...config};

        // Load the mesh
        this._ready = SceneLoader
            .ImportMeshAsync("", "./", loadModel(this.config.model), this.scene)
            .then(({meshes}) => {
                this.onMeshLoaded(meshes);
                return this;
            });
    }

    onMeshLoaded(newMeshes) {
        // Base mesh setup
        this.mesh = newMeshes[0];
        this.mesh.scaling.x = this.config.scaling;
        this.mesh.scaling.y = this.config.scaling;
        this.mesh.scaling.z = -this.config.scaling;

        // Material
        this.material = new StandardMaterial(`material_${this.id}`, this.scene);
        if (this.config.color) {
            this.material.diffuseColor = getBabylonColor3(this.config.color);
            this.material.specularColor = getBabylonColor3([0.3, 0.3, 0.3]);
        }
        this.mesh.material = this.material;

        // Position & orientation
        this.setOrientation(this.config.orientation);
        this.setPosition(this.config.position);
        this.scene.shadowGenerator.addShadowCaster(this.mesh);

        // Create FOV visualization (floor sector)
        const radius = this.config.fov_radius * this.config.scaling;
        // Angle in radians for rotation offset
        const fovAngleRad = this.config.fov_angle_deg * Math.PI / 180;
        // Fraction of full circle for the disc arc
        const arcFraction = this.config.fov_angle_deg / 360;

        const fovMaterial = new StandardMaterial(`fovMat_${this.id}`, this.scene);
        fovMaterial.diffuseColor = getBabylonColor3(this.config.color);
        fovMaterial.emissiveColor = getBabylonColor3(this.config.color).scale(0.8);
        fovMaterial.specularColor = getBabylonColor3(this.config.color).scale(0.1);
        fovMaterial.alpha = 0.3;
        fovMaterial.backFaceCulling = false;

        this.fovDisc = MeshBuilder.CreateDisc(`fovDisc_${this.id}`, {
            radius: radius,
            tessellation: 64,
            arc: arcFraction
        }, this.scene);
        this.fovDisc.material = fovMaterial;
        this.fovDisc.parent = this.mesh;
        // Slightly above the floor to avoid z-fighting
        this.fovDisc.position = new Vector3(0, 0.001, 0);
        // Flatten on the ground and center the sector on forward axis
        this.fovDisc.rotation = new Vector3(
            Math.PI / 2,
            fovAngleRad / 2,
            0
        );

    }

    addText() {
        // ... unchanged text method
        const dynamicTexture = new DynamicTexture("DynamicTexture", {width: 256, height: 256}, this.scene);
        const ctx = dynamicTexture.getContext();
        ctx.clearRect(0, 0, 256, 256);
        ctx.beginPath();
        ctx.arc(128, 128, 120, 0, Math.PI * 2, false);
        ctx.fillStyle = 'transparent';
        ctx.fill();
        ctx.lineWidth = 20;
        ctx.strokeStyle = getHTMLColor(this.config.text_color);
        ctx.stroke();
        const text = this.config.text;
        const maxTextWidth = 200;
        let fontSize = 250;
        let font = `bold ${fontSize}px Arial`;
        ctx.font = font;
        let measuredWidth = ctx.measureText(text).width;
        while (measuredWidth > maxTextWidth && fontSize > 10) {
            fontSize -= 10;
            font = `bold ${fontSize}px Arial`;
            ctx.font = font;
            measuredWidth = ctx.measureText(text).width;
        }
        const yOffset = 128 + fontSize / 3;
        const [r, g, b] = this.config.text_color;
        const textColor = `rgb(${Math.floor(r * 255)}, ${Math.floor(g * 255)}, ${Math.floor(b * 255)})`;
        dynamicTexture.drawText(text, null, yOffset, font, textColor, null, true);
        dynamicTexture.update();
        const textMaterial = new StandardMaterial("textMaterial", this.scene);
        textMaterial.diffuseTexture = dynamicTexture;
        textMaterial.diffuseTexture.hasAlpha = true;
        textMaterial.useAlphaFromDiffuseTexture = true;
        textMaterial.backFaceCulling = false;
        textMaterial.emissiveColor = getBabylonColor3([0.01, 0.01, 0.01]);
        const textHolder = new TransformNode("textHolder", this.scene);
        textHolder.parent = this.mesh;
        textHolder.position = new Vector3(0.0325, 0.05, 0);
        textHolder.scaling = new Vector3(1, 1, -1);
        const textPlane = MeshBuilder.CreatePlane("textPlane", {width: 0.1, height: 0.1}, this.scene);
        textPlane.material = textMaterial;
        textPlane.parent = textHolder;
        textPlane.rotation = new Vector3(0, 1.57, 0);
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
        } else if (typeof position === 'object' && position !== null && 'x' in position) {
            coords = [position.x, position.y, position.z];
        } else {
            throw new Error('Invalid position format. Expected [x, y, z] or {x, y, z}.');
        }
        this.position = coords;
        this.mesh.position = coordinatesToBabylon([this.position[0], this.position[1], this.config.scaling * this.config.z_offset]);
    }

    setState(x, y, psi) {
        this.setPosition([x, y, 0]);
        const orientation = Quaternion.fromEulerAngles([psi, 0, 0], 'zyx', true);
        this.setOrientation(orientation);
    }

    update(data) {
        return undefined;
    }

    onLoaded() {
        return this._ready;
    }

    setVisibility(visible) {
        super.setVisibility(visible);
        if (this.fovDisc){
            this.fovDisc.isVisible = visible;
        }
    }
}
