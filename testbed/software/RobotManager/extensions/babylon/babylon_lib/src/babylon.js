import {Scene} from "./Scene";

import {
    ArcRotateCamera,
    Vector3,
    Color3,
    HemisphericLight,
    SpotLight,
    DirectionalLight,
    MeshBuilder,
    GlowLayer,
    Matrix,
    Scene as BabylonScene,
    ShadowGenerator, StandardMaterial, Engine, PointerEventTypes,
} from "@babylonjs/core";

import {FramingBehavior} from "@babylonjs/core/Behaviors/Cameras/framingBehavior";
import {AdvancedDynamicTexture, TextBlock, Control} from "@babylonjs/gui";
import {Websocket} from "./lib/websocket.js"
import {coordinatesToBabylon, getBabylonColor, getBabylonColor3, getHTMLColor} from "./lib/babylon_utils.js"
import {drawCoordinateSystem} from "./lib/objects/coordinate_system";
import {BabylonFloor, BabylonFloorInstanced, BabylonSimpleFloor} from "./lib/objects/floor.js";
import {BabylonBox} from "./lib/objects/box.js";
import {Quaternion} from "./lib/quaternion.js";
import {BabylonObject} from "./lib/objects.js";
import {BabylonBilbo} from "./lib/objects/bilbo.js";
import {sleep} from "./lib/utilities.js";
import {BabylonFrodo} from "./lib/objects/frodo.js";

const DEFAULT_CONFIG = {
    websocket_host: 'localhost',
    websocket_port: '9000',
    coordinate_system_length: 0.5,
    show_coordinate_system: true,
    camera: {
        position: new Vector3(0, 0, 1),
        target: new Vector3(0, 0, 0),
        alpha: 7.2,
        beta: 1,
        radius: 3.5,
    },
    lights: {
        hemispheric_direction: coordinatesToBabylon([2, 0, 1]),
    },

    colors: {
        // background: [0.8, 0.8, 0.8],
        background: [31 / 255, 32 / 255, 35 / 255],
    },
    ui: {
        text_color: [1, 1, 1],
        font_size: 40,
    }
}

// =================================================================================================================
export class Babylon extends Scene {
    constructor(canvasOrEngine, config = {}) {
        super(canvasOrEngine);

        this.config = {...DEFAULT_CONFIG, ...config};

        this.objects = {}

        this.canvas = canvasOrEngine;
        this.websocket = new Websocket({host: this.config.websocket_host, port: this.config.websocket_port});

        this.websocket.on('message', this.handleMessage.bind(this));  // also works

        this._addResizeListener();
        this.websocket.connect();
        this.createScene();
        this.addObjectPicker();
        this.createUI();
    }


    // -----------------------------------------------------------------------------------------------------------------
    _addResizeListener() {
        // const engine = new Engine(this.canvas, true, {preserveDrawingBuffer: true, stencil: true});
        const engine = this.scene.getEngine();
        // This ensures Babylon accounts for HiDPI / Retina displays
        engine.setHardwareScalingLevel(1 / window.devicePixelRatio);

        // Resize on window change
        window.addEventListener("resize", () => {
            engine.resize();
        });
    }

    // =================================================================================================================
    createScene() {
        const scene = this.scene;

        // — CAMERA —
        const camera = new ArcRotateCamera("Camera",
            this.config.camera.alpha, this.config.camera.beta,
            this.config.camera.radius, coordinatesToBabylon(this.config.camera.target), scene);

        // camera.setPosition(coordinatesToBabylon(this.config.camera.position));
        camera.attachControl(this.canvas, true);
        camera.inputs.attached.keyboard.detachControl();
        camera.wheelPrecision = 100;
        camera.minZ = 0.1;
        camera.lowerBetaLimit = 0.0;            // optional: avoid exactly straight-down
        camera.upperBetaLimit = Math.PI / 2 - 0.05;    // never go below the ground plane
        camera.lowerRadiusLimit = 0.5;
        camera.upperRadiusLimit = 5;
        this.camera = camera;

        //
        // this.framing = new FramingBehavior()
        // camera.addBehavior(this.framing);
        // this.framing.radiusScale = 1.2;       // padding around the mesh
        // this.framing.focusOnFramedObject = true;
        //
        // this.scene.animationsEnabled = true;
        // console.log(camera.getBehaviorByName("Framing"));  // should log your FramingBehavior instance



        // camera.onViewMatrixChangedObservable.add(() => {
        //     // serialize exactly the numbers you’ll want to hard‐code next time
        //     const cfg = {
        //         alpha: camera.alpha,
        //         beta: camera.beta,
        //         radius: camera.radius,
        //         target: [camera.target.x, camera.target.y, camera.target.z],
        //         position: [camera.position.x, camera.position.y, camera.position.z]
        //     };
        //     console.log("NEW CAMERA CONFIG:\n", JSON.stringify(cfg, null, 2));
        // });

        // — LIGHT —
        new HemisphericLight("light", coordinatesToBabylon(this.config.lights.hemispheric_direction), scene).intensity = 0.3;
        new GlowLayer("glow", scene).intensity = 0.1;

        // - SHADOW -
        this.dirLight = new DirectionalLight(
            "shadowLight",
            new Vector3(0, 0, 0),
            this.scene
        );
        this.dirLight.position = coordinatesToBabylon([1, 1, 10]);
        this.dirLight.direction = coordinatesToBabylon([1, 1, -1]);
        this.dirLight.intensity = 1;
        this.dirLight.shadowEnabled = true;


        this.shadowGen = new ShadowGenerator(1024, this.dirLight);
        this.scene.shadowGenerator = this.shadowGen;
        this.shadowGen.useExponentialShadowMap = true;
        this.shadowGen.depthScale = 200;                       // default is 50 — try 100–200

        this.shadowGen.useContactHardeningShadowMap = true;
        this.shadowGen.contactHardeningLightSizeUVRatio = 0.1;
        this.shadowGen.contactHardeningDarkeningLightSizeUVRatio = 0.3;
        this.shadowGen.setDarkness(0);

        this.dirLight2 = new DirectionalLight(
            "shadowLight2",
            new Vector3(0, 0, 0),
            this.scene
        );
        this.dirLight2.position = coordinatesToBabylon([1, -1, 10]);
        this.dirLight2.direction = coordinatesToBabylon([1, -1, -1]);
        this.dirLight2.intensity = 1;
        this.dirLight2.shadowEnabled = true;


        this.shadowGen2 = new ShadowGenerator(1024, this.dirLight2);
        this.scene.shadowGenerator2 = this.shadowGen2;
        this.shadowGen2.useExponentialShadowMap = true;
        this.shadowGen2.depthScale = 200;                       // default is 50 — try 100–200

        this.shadowGen2.useContactHardeningShadowMap = true;
        this.shadowGen2.contactHardeningLightSizeUVRatio = 0.1;
        this.shadowGen2.contactHardeningDarkeningLightSizeUVRatio = 0.3;
        this.shadowGen2.setDarkness(0);

        // — BACKGROUND —
        scene.clearColor = getBabylonColor(this.config.colors.background);

        scene.ambientColor = new Color3(0.3, 0.3, 0.3);

        scene.fogMode = BabylonScene.FOGMODE_EXP2;
        scene.fogDensity = 0.1;
        scene.fogColor = getBabylonColor3(this.config.colors.background);

        // — COORDINATES +
        if (this.config.show_coordinate_system) {
            drawCoordinateSystem(this.scene, this.config.coordinate_system_length);
        }

        this.addTestObjects();


        return scene;
    }

    // =================================================================================================================
    addObjectPicker() {
        this.scene.onPointerObservable.add((pointerInfo) => {
            if (pointerInfo.type === PointerEventTypes.POINTERDOWN) {
                this._handleSingleSceneClick();

            } else if (pointerInfo.type === PointerEventTypes.POINTERDOUBLETAP) {
                this._handleDoubleSceneClick();
            }
        });
    }

    // =================================================================================================================
    _handleSingleSceneClick() {
        const pick = this.scene.pick(this.scene.pointerX, this.scene.pointerY);
        if (pick.hit) {
            if (pick.pickedMesh.metadata) {
                for (const obj of Object.values(this.objects)) {
                    obj.highlight(false);
                }

                pick.pickedMesh.metadata.object.highlight(true);

                // const center = pick.pickedMesh.getBoundingInfo()
                //     .boundingBox
                //     .centerWorld;
                //
                // // point the camera at that position
                // this.camera.setTarget(center);


                // this.framing.zoomOnMesh(pick.pickedMesh, /* focusXZ=*/ true, () => {
                //     console.log("framing animation done");
                // });


            } else {
                console.log("No object metadata");
            }
        }
    }

    _handleDoubleSceneClick() {
        for (const obj of Object.values(this.objects)) {
            obj.highlight(false);
        }
    }

    // =================================================================================================================
    createUI() {
        // — UI TEXTBLOCKS —
        const ui = AdvancedDynamicTexture.CreateFullscreenUI("ui", true, this.scene);

        const makeText = (fontSize, hAlign, vAlign, padding) => {
            const tb = new TextBlock();
            tb.fontSize = `${fontSize}px`;
            tb.color = getHTMLColor(this.config.ui.text_color);
            tb.textHorizontalAlignment = hAlign;
            tb.textVerticalAlignment = vAlign;
            Object.entries(padding).forEach(([k, v]) => (tb[`padding${k}`] = `${v}px`));
            ui.addControl(tb);
            return tb;
        };

        this.textbox_time = makeText(this.config.ui.font_size, Control.HORIZONTAL_ALIGNMENT_LEFT, Control.VERTICAL_ALIGNMENT_TOP, {
            Top: 3,
            Left: 10
        });
        this.textbox_status = makeText(this.config.ui.font_size, Control.HORIZONTAL_ALIGNMENT_RIGHT, Control.VERTICAL_ALIGNMENT_TOP, {
            Top: 3,
            Right: 10
        });
        this.textbox_title = makeText(this.config.ui.font_size, Control.HORIZONTAL_ALIGNMENT_CENTER, Control.VERTICAL_ALIGNMENT_TOP, {
            Top: 3,
            Left: 3
        });

        this.textbox_title.text = "title"
        this.textbox_status.text = "status";
        this.textbox_time.text = "time";
    }

    // =================================================================================================================
    /**
     *
     * @param {BabylonObject} object
     */
    addObject(object) {

        // Check if the object id is already in objects
        if (object.id in this.objects) {
            console.warn(`Object with id ${object.id} already exists`);
            return;
        }
        this.objects[object.id] = object;
    }

    // =================================================================================================================
    handleMessage(msg) {
        console.log(`Message: ${msg}`);
    }

    // =================================================================================================================
    addTestObjects() {

        this.floor = new BabylonFloorInstanced('floor', this.scene,
            {tile_size: 1, tiles_x: 40, tiles_y: 40, border_type: null});


        this.wall1 = new BabylonBox("wall1", this.scene,
            {
                size: {x: 4, y: 0.05, z: 0.25},
                texture: 'wood4.png',
                texture_uscale: 10
            });
        this.wall1.setPosition([0, 2 + 0.025, 0.125]);

        this.wall2 = new BabylonBox("wall2", this.scene, {
            size: {x: 4, y: 0.05, z: 0.25}, texture: 'wood4.png',
            texture_uscale: 10
        });
        this.wall2.setPosition([0, -2 - 0.025, 0.125]);

        this.wall3 = new BabylonBox("wall3", this.scene, {
            size: {x: 4 + 0.1, y: 0.05, z: 0.25}, texture: 'wood4.png',
            texture_uscale: 10
        });
        this.wall3.setOrientation(Quaternion.fromAngleAxis(Math.PI / 2, [0, 0, 1]));
        this.wall3.setPosition([2 + 0.025, 0, 0.125]);

        this.wall4 = new BabylonBox("wall4", this.scene, {
            size: {x: 4 + 0.1, y: 0.05, z: 0.25}, texture: 'wood4.png',
            texture_uscale: 10
        });
        this.wall4.setOrientation(Quaternion.fromAngleAxis(Math.PI / 2, [0, 0, 1]));
        this.wall4.setPosition([-2 - 0.025, 0, 0.125]);


        this.bilbo1 = new BabylonBilbo('bilbo1', this.scene, {text: '1', color: [1, 0, 0]});

        this.bilbo1.onLoaded().then(() => {
            this.bilbo1.setState(-1, 1, Math.PI / 4, Math.PI / 4);
        })
        this.addObject(this.bilbo1);

        this.bilbo2 = new BabylonBilbo('bilbo2', this.scene, {text: '2', color: [0.5, 0.5, 0.7]});

        this.bilbo2.onLoaded().then(() => {
            this.bilbo2.setState(1, 0, 0, Math.PI);
        })

        this.addObject(this.bilbo2);


        this.frodo1 = new BabylonFrodo('frodo1', this.scene, {scaling: 1});

        this.frodo1.onLoaded().then(() => {
            this.frodo1.setState(0, 1, -Math.PI / 4);
        })

        this.frodo2 = new BabylonFrodo('frodo2', this.scene, {scaling: 1, color: [0, 0.7, 0.2]});
        this.frodo2.onLoaded().then(() => {
            this.frodo2.setState(0, -1, 0);
        })

    }
}
