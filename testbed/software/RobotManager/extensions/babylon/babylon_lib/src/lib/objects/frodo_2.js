import {BabylonObject} from "../objects.js";
import {
    CreateBox,
    StandardMaterial,
    Texture,
    SceneLoader,
    DynamicTexture,
    TransformNode,
    Vector3,
    MeshBuilder
} from "@babylonjs/core";
import {
    coordinatesToBabylon,
    getBabylonColor,
    getBabylonColor3,
    loadTexture,
    loadModel,
    getHTMLColor,
} from "../babylon_utils.js";
import {Quaternion} from '../quaternion.js'


export class BabylonFrodo extends BabylonObject {
    constructor(id, scene, config = {}) {
        super(id, scene, config);

        const default_config = {
            model: 'frodo_generic.babylon',
            text: '',
            text_color: [1, 1, 1],
            color: [0.4, 0.4, 0.4],
            scaling: 1,
            position: [0, 0, 0],
            orientation: [1, 0, 0, 0],
            z_offset: 0,
            fov_radius: 1,
            fov_angle_deg: 120,
        }

        this.config = {...default_config, ...config};

        console.log(loadModel(this.config.model));

        // Load the mesh
        this._ready = SceneLoader
            .ImportMeshAsync("", "./", loadModel(this.config.model), this.scene)
            .then(({meshes}) => {
                this.onMeshLoaded(meshes, /*…*/)
                return this;            // resolve with `this` so callers can chain
            });

        // Now we wait for the mesh to be loaded. The rest of the configuration resumes in onMeshLoaded()

    }

    onMeshLoaded(newMeshes, particleSystems, skeletons) {
        console.log("onMeshLoaded");

        // Mesh
        this.mesh = newMeshes[0];
        this.mesh.scaling.x = this.config.scaling;
        this.mesh.scaling.y = this.config.scaling;
        this.mesh.scaling.z = -this.config.scaling;

        // Material
        this.material = new StandardMaterial("material", this.scene);
        if (this.config.color) {
            this.material.diffuseColor = getBabylonColor3(this.config.color);
            this.material.specularColor = getBabylonColor3([0.3, 0.3, 0.3]);
        }
        this.mesh.material = this.material;

        // if (this.config.text) {
        //     this.addText();
        // }

        // this.mesh.receiveShadows = true;

        // Set Position + Orientation
        this.setOrientation(this.config.orientation);
        this.setPosition(this.config.position);

        this.scene.shadowGenerator.addShadowCaster(this.mesh);

    }

    addText() {
        //
        // Create a dynamic texture with a larger size.
        const dynamicTexture = new DynamicTexture("DynamicTexture", {width: 256, height: 256}, this.scene);
        const ctx = dynamicTexture.getContext();
        ctx.clearRect(0, 0, 256, 256);
        //
        // Draw a circle
        ctx.beginPath();
        ctx.arc(128, 128, 120, 0, Math.PI * 2, false); // Center (128,128) with a radius of 120.
        ctx.fillStyle = 'transparent';
        ctx.fill();
        ctx.lineWidth = 20;
        ctx.strokeStyle = getHTMLColor(this.config.text_color);
        ctx.stroke();
        //
        // The text to display.
        const text = this.config.text; // Change this to any text.

        // Define the maximum allowed text width (in pixels) within the circle.
        const maxTextWidth = 200;
        let fontSize = 250; // Start with a large font size.
        let font = `bold ${fontSize}px Arial`;

        // Set the context font and measure the text width.
        ctx.font = font;
        let measuredWidth = ctx.measureText(text).width;

        // Reduce the font size until the text fits within the maxTextWidth.
        while (measuredWidth > maxTextWidth && fontSize > 10) {
            fontSize -= 10;
            font = `bold ${fontSize}px Arial`;
            ctx.font = font;
            measuredWidth = ctx.measureText(text).width;
        }

        // Compute a y-offset to vertically center the text in the circle.
        // This calculation can be adjusted based on your design needs.
        const yOffset = 128 + fontSize / 3;

        // Draw the text on top of the circle without clearing the canvas.
        // Passing null as clearColor ensures our circle remains.
        const brightnessFactor = 1;
        const [r, g, b] = this.config.text_color;
        const textColor = `rgb(${Math.floor(r * 255 * brightnessFactor)}, ${Math.floor(g * 255 * brightnessFactor)}, ${Math.floor(b * 255 * brightnessFactor)})`;


        dynamicTexture.drawText(text, null, yOffset, font, textColor, null, true);
        dynamicTexture.update();

        // Create a new material using the dynamic texture.
        const textMaterial = new StandardMaterial("textMaterial", this.scene);
        textMaterial.diffuseTexture = dynamicTexture;
        textMaterial.diffuseTexture.hasAlpha = true;
        textMaterial.useAlphaFromDiffuseTexture = true;
        // textMaterial.disableLighting = true;
        textMaterial.backFaceCulling = false;

        textMaterial.emissiveColor = getBabylonColor3([0.01, 0.01, 0.01]);

        // Create an intermediate transform node to counteract the parent's negative scaling.
        const textHolder = new TransformNode("textHolder", this.scene);
        textHolder.parent = this.mesh;
        textHolder.position = new Vector3(0.0325, 0.05, 0);
        textHolder.scaling = new Vector3(1, 1, -1);
        //
        // Create the plane that will display the text.
        const textPlane = MeshBuilder.CreatePlane("textPlane", {width: 0.1, height: 0.1}, this.scene);
        textPlane.material = textMaterial;
        textPlane.parent = textHolder;
        textPlane.rotation = new Vector3(0, 1.57, 0);
        //
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
        this.mesh.position = coordinatesToBabylon([this.position[0], this.position[1], this.config.scaling * this.config.z_offset]);
    }

    setState(x, y, psi) {
        this.setPosition([x, y, 0])
        const orientation = Quaternion.fromEulerAngles([psi, 0, 0], 'zyx', true);
        this.setOrientation(orientation);
    }

    update(data) {
        return undefined;
    }

    onLoaded() {
        return this._ready;
    }
}