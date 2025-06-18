import {CreateLines} from "@babylonjs/core";
import {coordinatesToBabylon, getBabylonColor3} from "../babylon_utils";

export function drawCoordinateSystem(scene, length) {
    const z_offset = 0.001;
    const points_x = [
        coordinatesToBabylon([0, 0, z_offset]),
        coordinatesToBabylon([length, 0, z_offset])
    ]
    const points_y = [
        coordinatesToBabylon([0, 0, z_offset]),
        coordinatesToBabylon([0, length, z_offset])
    ]
    const points_z = [
        coordinatesToBabylon([0, 0, z_offset]),
        coordinatesToBabylon([0, 0, length + z_offset])
    ]
    const line_x = CreateLines("line_x", {points: points_x}, scene);
    line_x.color = getBabylonColor3([1, 0, 0]);

    const line_y = CreateLines("line_y", {points: points_y}, scene);
    line_y.color = getBabylonColor3([0, 1, 0]);

    const line_z = CreateLines("line_z", {points: points_z}, scene);
    line_z.color = getBabylonColor3([0, 0, 1]);
}