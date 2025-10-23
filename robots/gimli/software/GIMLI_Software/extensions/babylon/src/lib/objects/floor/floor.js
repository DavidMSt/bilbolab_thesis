import {BabylonObject} from '../../objects.js';
import {adjustMaterialBrightness, getBabylonColor3, loadTexture} from "../../babylon_utils.js";
import {
    CreateTiledGround,
    StandardMaterial,
    Texture,
    MultiMaterial,
    SubMesh,
    MeshBuilder,
    Vector3, Color3,
    DynamicTexture
} from "@babylonjs/core";

// =====================================================================================================================
// export class BabylonFloor extends BabylonObject {
//     constructor(id, scene, config = {}) {
//         super(id, scene, config);
//
//         const default_config = {
//             tile_size: 0.5,
//             tiles_x: 10,
//             tiles_y: 10,
//             color1: [0.5, 0.5, 0.5],
//             color2: [0.65, 0.65, 0.65],
//             texture_1: 'floor_bright.png',
//             texture_2: 'floor_bright.png',
//             brightness_1: 1,
//             brightness_2: 0.9,
//             border_type: 'line', // Can be null, 'line' or 'tile'
//             border_color: [0.4, 0.4, 0.4],
//             border_width: 0.025,
//             border_texture: 'floor_bright.png',
//             border_texture_brightness: 0.6,
//
//             show_coordinates: true,
//             coordinates_font_size: 12,
//             coordinates_color: [0, 0, 0]
//         };
//
//         this.config = {...default_config, ...config};
//
//         // Create the tiled ground mesh
//         this.mesh = new CreateTiledGround("Tiled Ground", {
//             xmin: -this.config.tile_size * this.config.tiles_x / 2,
//             zmin: -this.config.tile_size * this.config.tiles_y / 2,
//             xmax: this.config.tile_size * this.config.tiles_x / 2,
//             zmax: this.config.tile_size * this.config.tiles_y / 2,
//             subdivisions: {
//                 h: this.config.tiles_y,
//                 w: this.config.tiles_x
//             }
//         }, this.scene);
//
//         // material 1
//         this.material_1 = new StandardMaterial("material_1", this.scene);
//         if (this.config.texture_1) {
//             const tex1 = loadTexture(this.config.texture_1);
//             this.material_1.diffuseTexture = new Texture(tex1, this.scene);
//         } else {
//             this.material_1.diffuseColor = getBabylonColor3(this.config.color1);
//         }
//         this.material_1.specularColor = getBabylonColor3([0, 0, 0]);
//
//         // material 2
//         this.material_2 = new StandardMaterial("material_2", this.scene);
//         if (this.config.texture_2) {
//             const tex2 = loadTexture(this.config.texture_2);
//             this.material_2.diffuseTexture = new Texture(tex2, this.scene);
//         } else {
//             this.material_2.diffuseColor = getBabylonColor3(this.config.color2);
//         }
//         this.material_2.specularColor = getBabylonColor3([0, 0, 0]);
//
//
//         this.material_1.emissiveColor = getBabylonColor3([0, 0, 0]);
//         this.material_2.emissiveColor = getBabylonColor3([0, 0, 0]);
//
//         // Combine into MultiMaterial
//         this.multimat = new MultiMaterial("multi", this.scene);
//         this.multimat.subMaterials.push(this.material_1);
//         this.multimat.subMaterials.push(this.material_2);
//
//         // If using tile-border mode, add a third material for the border tiles
//         if (this.config.border_type === 'tile') {
//             this.border_material = new StandardMaterial("border_material", this.scene);
//             if (this.config.border_texture) {
//                 const tex = loadTexture(this.config.border_texture);
//                 this.border_material.diffuseTexture = new Texture(tex, this.scene);
//             } else {
//                 this.border_material.diffuseColor = getBabylonColor3(this.config.border_color);
//             }
//             this.border_material.specularColor = getBabylonColor3([0, 0, 0]);
//             this.multimat.subMaterials.push(this.border_material);
//         }
//
//         this.mesh.material = this.multimat;
//
//         // Split the mesh into checkerboard (or border) subMeshes
//         this.verticesCount = this.mesh.getTotalVertices();
//         this.tileIndicesLength = this.mesh.getIndices().length / (this.config.tiles_x * this.config.tiles_y);
//         this.mesh.subMeshes = [];
//
//         let base = 0;
//         for (let row = 0; row < this.config.tiles_y; row++) {
//             for (let col = 0; col < this.config.tiles_x; col++) {
//                 const isBorderTile = row === 0 || row === this.config.tiles_y - 1 || col === 0 || col === this.config.tiles_x - 1;
//                 let matIndex;
//                 if (this.config.border_type === 'tile' && isBorderTile) {
//                     // use the last material (border)
//                     matIndex = this.multimat.subMaterials.length - 1;
//                 } else {
//                     // standard checker pattern
//                     matIndex = (row % 2) ^ (col % 2);
//                 }
//                 let submesh = new SubMesh(
//                     matIndex,
//                     0,
//                     this.verticesCount,
//                     base,
//                     this.tileIndicesLength,
//                     this.mesh
//                 )
//                 // submesh.receiveShadows = true;
//                 this.mesh.subMeshes.push(submesh);
//                 base += this.tileIndicesLength;
//             }
//         }
//
//         // If line-border mode, add a tube around the outer edge
//         if (this.config.border_type === 'line') {
//             const halfX = this.config.tile_size * this.config.tiles_x / 2;
//             const halfZ = this.config.tile_size * this.config.tiles_y / 2;
//             const y = 0; // lift slightly to avoid z-fighting
//             const path = [
//                 new Vector3(-halfX, y, -halfZ),
//                 new Vector3(halfX, y, -halfZ),
//                 new Vector3(halfX, y, halfZ),
//                 new Vector3(-halfX, y, halfZ),
//                 new Vector3(-halfX, y, -halfZ)
//             ];
//             const borderTube = MeshBuilder.CreateTube("border_tube", {
//                 path,
//                 radius: this.config.border_width / 2,
//                 sideOrientation: MeshBuilder.DOUBLESIDE,
//                 updatable: false
//             }, this.scene);
//             const borderLineMat = new StandardMaterial("border_line_material", this.scene);
//             borderLineMat.diffuseColor = getBabylonColor3(this.config.border_color);
//             borderLineMat.emissiveColor = getBabylonColor3(this.config.border_color);
//             borderLineMat.specularColor = getBabylonColor3([0, 0, 0]);
//             borderTube.material = borderLineMat;
//         }
//
//         this.mesh.receiveShadows = true;
//     }
//
//     highlight(state) {
//         // implement as needed
//     }
//
//     onMessage(message) {
//         // implement as needed
//     }
//
//     setOrientation(orientation) {
//         // implement as needed
//     }
//
//     setPosition(position) {
//         // implement as needed
//     }
//
//     update(data) {
//         // implement as needed
//     }
// }

// =====================================================================================================================
export class BabylonSimpleFloor extends BabylonObject {
    constructor(id, scene, payload = {}) {
        super(id, scene, payload);

        const defaultConfig = {
            size_x: 5,          // total width of the floor (X axis, meters or units)
            size_y: 5,          // total height/depth of the floor (Z axis)
            tile_size: 0.5,     // desired size of a single tile (square)
            texture: 'floor_bright.png', // URL or path to a texture image
            color: [0.7, 0.7, 0.7],      // fallback diffuse color
        };

        this.config = {...defaultConfig, ...this.config};

        // Compute how many tiles we want along each axis.
        // We round to the nearest integer so tiles align nicely with size.
        const tilesX = Math.max(1, Math.round(this.config.size_x / this.config.tile_size));
        const tilesY = Math.max(1, Math.round(this.config.size_y / this.config.tile_size));

        // NOTE: In Babylon, CreateGround's `subdivisions` only affects geometry density,
        // not UVs/texture repetition. We still set it based on the max tile count so
        // the mesh is reasonably dense (helps with lighting/shadows).
        const subdivisions = Math.max(tilesX, tilesY);

        // Create a simple ground mesh
        this.mesh = MeshBuilder.CreateGround(
            id + '_ground',
            {
                width: this.config.size_x,
                height: this.config.size_y,
                subdivisions,       // geometry density only; does NOT control texture tiling
                updatable: false,
            },
            this.scene
        );

        // Create material
        const mat = new StandardMaterial(id + '_mat', this.scene);
        mat.specularColor = new Color3(0, 0, 0);              // no specular highlights
        mat.emissiveColor = new Color3(0, 0, 0);              // keep shadows visible
        // (Backface culling on by default; fine for a ground plane)
        // mat.backFaceCulling = true;

        if (this.config.texture) {
            const texUrl = loadTexture(this.config.texture);
            const tex = new Texture(texUrl, this.scene);
            // Ensure tiling mode is wrapping (repeat)
            tex.wrapU = Texture.WRAP_ADDRESSMODE;
            tex.wrapV = Texture.WRAP_ADDRESSMODE;

            // **This is the key:** tile the texture across the ground by scaling UVs.
            // One repeat per tile along each axis.
            tex.uScale = tilesX;
            tex.vScale = tilesY;

            // Optional: if your texture looks vertically flipped, uncomment the next line:
            // tex.vScale = -tilesY;

            mat.diffuseTexture = tex;
        } else {
            // Fallback solid color if no texture provided
            mat.diffuseColor = getBabylonColor3(this.config.color);
        }

        this.mesh.material = mat;
        this.mesh.isPickable = false;

        // Receive shadows
        this.mesh.receiveShadows = true;

        // Store a couple of things in case you want to query/update later
        this._tilesX = tilesX;
        this._tilesY = tilesY;
    }

    highlight(state) {
        // optional highlight implementation
    }

    onMessage(message) {
        // optional message handling
    }

    setOrientation(orientation) {
        // optional orientation handling
    }

    setPosition(position) {
        if (Array.isArray(position)) {
            this.mesh.position = new Vector3(...position);
        } else if (position instanceof Vector3) {
            this.mesh.position = position;
        }
    }

    update(data) {
        // optional update implementation
        // If you ever add dynamic resizing, remember:
        // - Recompute tilesX/tilesY from size_x/size_y/tile_size
        // - Update mat.diffuseTexture.uScale / vScale
        // - Rebuild the ground if width/height change (or create as updatable)
    }

    delete() {
        this.mesh.dispose();
    }

    dim(state) {
        return undefined;
    }
}

// =====================================================================================================================
export class BabylonFloorInstanced extends BabylonObject {
    constructor(id, scene, payload = {}) {
        super(id, scene, payload);

        const defaultConfig = {
            tile_size: 0.5,
            tiles_x: 10,
            tiles_y: 10,
            color1: [0.5, 0.5, 0.5],
            color2: [0.65, 0.65, 0.65],
            texture_1: 'drawing_board.png',
            texture_2: 'drawing_board.png',
            brightness_1: 1,
            brightness_2: 0.9,
            border_type: 'line', // null, 'line', or 'tile'
            border_color: [0.4, 0.4, 0.4],
            border_width: 0.025,
            border_texture: 'floor_bright.png',
            border_texture_brightness: 0.6
        };

        this.config = {...defaultConfig, ...this.config};
        const {tile_size, tiles_x, tiles_y} = this.config;

        // --- Create materials ---
        this.material1 = new StandardMaterial(id + '_mat1', scene);
        if (this.config.texture_1) {
            const tex1 = loadTexture(this.config.texture_1);
            this.material1.diffuseTexture = new Texture(tex1, scene);
        } else {
            this.material1.diffuseColor = getBabylonColor3(this.config.color1);
        }
        this.material1.specularColor = getBabylonColor3([0, 0, 0]);
        this.material1.emissiveColor = getBabylonColor3([0, 0, 0]);

        this.material2 = new StandardMaterial(id + '_mat2', scene);
        if (this.config.texture_2) {
            const tex2 = loadTexture(this.config.texture_2);
            this.material2.diffuseTexture = new Texture(tex2, scene);
        } else {
            this.material2.diffuseColor = getBabylonColor3(this.config.color2);
        }
        this.material2.specularColor = getBabylonColor3([0, 0, 0]);
        this.material2.emissiveColor = getBabylonColor3([0, 0, 0]);

        if (this.config.border_type === 'tile') {
            this.borderMaterial = new StandardMaterial(id + '_mat_border', scene);
            if (this.config.border_texture) {
                const texB = loadTexture(this.config.border_texture);
                this.borderMaterial.diffuseTexture = new Texture(texB, scene);
            } else {
                this.borderMaterial.diffuseColor = getBabylonColor3(this.config.border_color);
            }
            this.borderMaterial.specularColor = getBabylonColor3([0, 0, 0]);
            this.borderMaterial.emissiveColor = getBabylonColor3([0, 0, 0]);
        }

        // --- Create invisible base tile meshes (flat on XZ plane) ---
        this.baseTiles = {};
        const createBase = (name, material) => {
            const mesh = MeshBuilder.CreateGround(
                id + '_' + name,
                {width: tile_size, height: tile_size},
                scene
            );
            mesh.material = material;
            mesh.isVisible = false;
            mesh.receiveShadows = true;
            mesh.isPickable = false;
            return mesh;
        };

        this.baseTiles.mat1 = createBase('template1', this.material1);
        this.baseTiles.mat2 = createBase('template2', this.material2);
        if (this.config.border_type === 'tile') {
            this.baseTiles.border = createBase('templateBorder', this.borderMaterial);
        }

        // --- Instance tiles in a grid ---
        this.tiles = [];
        for (let r = 0; r < tiles_y; r++) {
            this.tiles[r] = [];
            for (let c = 0; c < tiles_x; c++) {
                const isBorder = r === 0 || r === tiles_y - 1 || c === 0 || c === tiles_x - 1;
                let sourceMesh;
                if (this.config.border_type === 'tile' && isBorder) {
                    sourceMesh = this.baseTiles.border;
                } else {
                    sourceMesh = ((r + c) % 2 === 0)
                        ? this.baseTiles.mat1
                        : this.baseTiles.mat2;
                }

                const inst = sourceMesh.createInstance(id + `_tile_${c}_${r}`);
                // Map grid x->world x, grid y->world -z
                inst.isPickable = false;
                inst.position.x = (c - tiles_x / 2 + 0.5) * tile_size;
                inst.position.z = -(r - tiles_y / 2 + 0.5) * tile_size;
                this.tiles[r][c] = inst;
            }
        }

        // --- Optional line border ---
        if (this.config.border_type === 'line') {
            const halfX = tile_size * tiles_x / 2;
            const halfZ = tile_size * tiles_y / 2;
            const path = [
                new Vector3(-halfX, 0.01, -halfZ),
                new Vector3(halfX, 0.01, -halfZ),
                new Vector3(halfX, 0.01, halfZ),
                new Vector3(-halfX, 0.01, halfZ),
                new Vector3(-halfX, 0.01, -halfZ)
            ];
            const borderTube = MeshBuilder.CreateTube(
                id + '_borderTube',
                {path, radius: this.config.border_width / 2, sideOrientation: MeshBuilder.DOUBLESIDE},
                scene
            );
            const tubeMat = new StandardMaterial(id + '_borderLineMat', scene);
            tubeMat.diffuseColor = getBabylonColor3(this.config.border_color);
            tubeMat.emissiveColor = getBabylonColor3(this.config.border_color);
            tubeMat.specularColor = getBabylonColor3([0, 0, 0]);
            borderTube.material = tubeMat;
        }
    }

    /**
     * Toggle visibility of the tile at grid coords (x,y).
     * Here y in grid maps to -z in world.
     * @param {number} x - Column (0..tiles_x-1)
     * @param {number} y - Row (0..tiles_y-1)
     * @param {boolean} state - true = visible, false = hidden
     */
    setVisibility(x, y, state) {
        if (y < 0 || y >= this.tiles.length) return;
        const row = this.tiles[y];
        if (!row || x < 0 || x >= row.length) return;
        row[x].isVisible = state;
    }

    highlight(state) {
        return undefined;
    }

    onMessage(message) {
        return undefined;
    }

    setOrientation(orientation) {
        return undefined;
    }

    setPosition(position) {
        return undefined;
    }

    update(data) {
        return undefined;
    }

    dim(state) {
        return undefined;
    }

    delete() {
        this.mesh.dispose();
    }
}


