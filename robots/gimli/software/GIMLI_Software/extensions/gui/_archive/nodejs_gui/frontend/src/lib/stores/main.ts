import { writable } from "svelte/store";

export const currentBot = writable("");
export const activeBots = writable<string[]>([]);
export const currentView = writable("overview");

export const fps = writable(0);




currentBot.subscribe((bot) => {
    localStorage.setItem("currentBot", bot);
});

// load current bot from local storage on storage event
window.addEventListener("storage", (event) => {
    if (event.key === "currentBot") {
        currentBot.set(event.newValue);
    }
});

let lastFrameTime = Date.now();

function updateFps() {
    const now = Date.now();
    const delta = now - lastFrameTime;
    lastFrameTime = now;

    let currentfps = 1000 / delta;
    fps.update((fpsc) => {
        currentfps = currentfps > fpsc ? fpsc * 0.99 + currentfps * 0.01 : fpsc * 0.1 + currentfps * 0.9;
        return currentfps;
    });
    requestAnimationFrame(updateFps);
}

requestAnimationFrame(updateFps);