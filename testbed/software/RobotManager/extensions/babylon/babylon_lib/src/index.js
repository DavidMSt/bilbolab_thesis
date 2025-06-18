import {Babylon} from "./babylon";
import {Engine} from "@babylonjs/core";

// once the DOM is ready, kick everything off
window.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("renderCanvas");
    // an empty config; you can pass your own here
    new Babylon(canvas, {});



});
