// a thin wrapper around Babylon’s Engine + Scene setup
import { Engine, Scene as BScene } from "@babylonjs/core";

export class Scene {
  constructor(canvasOrEngine) {
    this.engineIsShared = canvasOrEngine instanceof Engine;
    if (this.engineIsShared) {
      this.engine = canvasOrEngine;
      this.canvas = this.engine.getRenderingCanvas();
    } else {
      this.canvas = typeof canvasOrEngine === "string"
        ? document.getElementById(canvasOrEngine)
        : canvasOrEngine;
      this.engine = new Engine(this.canvas, true);
    }

    this.scene = new BScene(this.engine);
    this.scene.useRightHandedSystem = true;

    if (!this.engineIsShared) {
      this.engine.runRenderLoop(() => this.scene.render());
      window.addEventListener("resize", () => this.engine.resize());
    }
  }
}
