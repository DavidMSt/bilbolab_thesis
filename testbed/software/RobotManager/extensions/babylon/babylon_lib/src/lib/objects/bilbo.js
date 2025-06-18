import {BabylonObject} from "../objects.js";
import {
    CreateBox,
    StandardMaterial,
    Texture,
    SceneLoader,
    DynamicTexture,
    TransformNode,
    Vector3,
    Mesh,
    MeshBuilder, Color3, ActionManager, ExecuteCodeAction
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
import {GlowLayer} from "@babylonjs/core/Layers/glowLayer";
import {HighlightLayer} from "@babylonjs/core/Layers/highlightLayer";

export class BabylonBilbo extends BabylonObject {
    constructor(id, scene, config = {}) {
        super(id, scene, config);

        const default_config = {
            model: 'bilbo_detail.babylon',
            static_rotation: Quaternion.fromEulerAngles([Math.PI/2, -Math.PI/2, 0], 'xyz', true),
            text: '',
            text_color: [1, 1, 1],
            color: [0.4, 0.4, 0.4],
            scaling: 1,
            model_scaling: 0.001,
            position: [0, 0, 0],
            orientation: [1, 0, 0, 0],
            z_offset: 0.125 / 2,
            dim: false,
        }

        this.config = {...default_config, ...config};

        this.static_rotation_quaternion = new Quaternion(this.config.static_rotation);

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
        // Mesh
        this.mesh = newMeshes[0];

        newMeshes.forEach(m => {
            m.metadata = m.metadata || {};
            m.metadata.object = this;
            m.isPickable = true;
        });

        this.mesh.scaling.x = this.config.scaling*this.config.model_scaling;
        this.mesh.scaling.y = this.config.scaling*this.config.model_scaling;
        this.mesh.scaling.z = -this.config.scaling*this.config.model_scaling;

        // Material
        this.material = new StandardMaterial("material", this.scene);
        // if (this.config.color) {
        //     this.material.diffuseColor = getBabylonColor3(this.config.color);
        //     this.material.specularColor = getBabylonColor3([0.3, 0.3, 0.3]);
        // }
        this.mesh.material = this.material;

        if (this.config.text) {
            this.addText();
        }

        // this.dim(this.config.dim);

        this._isHighlighted = false;
        this.highlight(this._isHighlighted);

        // Set Position + Orientation
        this.setOrientation(this.config.orientation);
        this.setPosition(this.config.position);


        this.scene.shadowGenerator.addShadowCaster(this.mesh);
        this.scene.shadowGenerator2.addShadowCaster(this.mesh);

        this.mesh.isPickable = true;

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
        this.textHolder = new TransformNode("textHolder", this.scene);
        this.textHolder.parent = this.mesh;
        this.textHolder.position = new Vector3(0.0325, 0.05, 0);
        this.textHolder.scaling = new Vector3(1, 1, -1);
        //
        // Create the plane that will display the text.
        this.textPlane = MeshBuilder.CreatePlane("textPlane", {width: 0.1, height: 0.1}, this.scene);
        this.textPlane.material = textMaterial;
        this.textPlane.parent = this.textHolder;
        this.textPlane.rotation = new Vector3(0, 1.57, 0);

        this.textPlane.metadata = this.textPlane.metadata || {};
        this.textPlane.metadata.object = this;
        this.textPlane.isPickable = true;
        //
    }

    onMessage(message) {
        return undefined;
    }

    setOrientation(orientation) {
        this.orientation = new Quaternion(orientation);
        this.mesh.rotationQuaternion = this.orientation.multiply(this.static_rotation_quaternion).babylon();
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

    setState(x, y, theta, psi) {
        this.setPosition([x, y, 0])
        const orientation = Quaternion.fromEulerAngles([psi, theta, 0], 'zyx', true);
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

        if (this.textHolder) {
            this.textHolder.isVisible = visible;
        }
        if (this.textPlane) {
            this.textPlane.isVisible = visible;
        }
    }

    highlight(state) {

        this._isHighlighted = state;

        if (state) {
            // 1) create a ring plane if needed
            if (!this._highlightPlane) {
                // compute size from the bounding sphere
                const bs = this.mesh.getBoundingInfo().boundingSphere;
                const radius = bs.radiusWorld * 1.2;
                const diameter = radius * 2;

                // draw ring on a DynamicTexture with RGBA fill
                const texSize = 512;
                const dt = new DynamicTexture(`hlTex_${this.id}`, {width: texSize, height: texSize}, this.scene, false);
                const ctx = dt.getContext();
                ctx.clearRect(0, 0, texSize, texSize);

                const c = texSize / 2;
                const thickness = texSize * 0.1;

                ctx.beginPath();
                // outer circle
                ctx.arc(c, c, c, 0, Math.PI * 2);
                // inner “cut‐out”
                ctx.arc(c, c, c - thickness, 0, Math.PI * 2, true);

                // use RGBA with 30% opacity (change 0.3 to taste)
                const [r, g, b] = this.config.color;
                ctx.fillStyle = `rgba(${r * 255}, ${g * 255}, ${b * 255}, 0.5)`;
                ctx.fill();
                dt.update();

                // build an alpha-blend emissive material
                const mat = new StandardMaterial(`hlMat_${this.id}`, this.scene);
                mat.diffuseTexture = dt;
                mat.diffuseTexture.hasAlpha = true;
                mat.useAlphaFromDiffuseTexture = true;
                mat.transparencyMode = StandardMaterial.MATERIAL_ALPHABLEND;
                mat.backFaceCulling = false;
                mat.specularColor = new Color3(0, 0, 0);
                mat.emissiveColor = getBabylonColor3(this.config.color);
                // ensure correct depth sorting with other transparent objects
                mat.needDepthPrePass = true;

                // create the plane and position it just under the robot
                const plane = MeshBuilder.CreatePlane(`hlPlane_${this.id}`, {size: diameter}, this.scene);
                plane.material = mat;
                plane.rotation.x = Math.PI / 2;
                plane.isPickable = false;

                // position plane at robot’s feet
                plane.position = this.mesh.position.clone();
                plane.position.y -= this.config.scaling * this.config.z_offset;
                plane.position.y += 0.0001;

                this._highlightPlane = plane;
            }

            // 2) show & follow the mesh
            this._highlightPlane.isVisible = true;
            this._highlightPlane.position.x = this.mesh.position.x;
            this._highlightPlane.position.z = this.mesh.position.z;

        } else {
            // hide when state=false
            if (this._highlightPlane) {
                this._highlightPlane.isVisible = false;
            }
        }
    }

    /**
     * Make the model translucent when `state` is true, opaque when false.
     * @param {boolean} state — true to dim (translucent), false to restore.
     */

    /**
     * Fade the model (and its text) in/out.
     * @param {boolean} state — true = dim (30% opacity), false = full (100%).
     */
    dim(state) {
        const alpha = state ? 0.5 : 1.0;

        // Gather the root mesh plus any imported sub-meshes
        const parts = [this.mesh];
        if (this.mesh.getChildMeshes) {
            parts.push(...this.mesh.getChildMeshes());
        }

        for (const m of parts) {
            const mat = m.material;
            if (!mat || !(mat instanceof StandardMaterial)) {
                continue;
            }

            mat.alpha = alpha;

            // If this material is using a diffuse-texture alpha (i.e. your text)
            if (mat.useAlphaFromDiffuseTexture && mat.diffuseTexture) {
                // always leave in alpha-blend so its mask shows
                mat.transparencyMode = StandardMaterial.MATERIAL_ALPHABLEND;
                mat.backFaceCulling = false;
                mat.needDepthPrePass = state;
            } else {
                // robot body parts
                if (alpha < 1) {
                    mat.transparencyMode = StandardMaterial.MATERIAL_ALPHABLEND;
                    mat.backFaceCulling = false;
                    mat.needDepthPrePass = true;
                } else {
                    mat.transparencyMode = StandardMaterial.MATERIAL_OPAQUE;
                    mat.backFaceCulling = true;
                    mat.needDepthPrePass = false;
                }
            }
        }
    }
}