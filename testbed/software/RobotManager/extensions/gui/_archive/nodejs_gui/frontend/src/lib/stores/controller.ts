import { writable } from "svelte/store";
import {sendMessage} from "./messages";

export const controller = writable([]);
//     [{
//     id: "3240234234324324",
//     name: "controller1",
//     assignedBot: null,
// },
// {
//     id: "32402334234324324",
//     name: "controller2",
//     assignedBot: null,
// },
// {
//     id: "32402234234324324",
//     name: "controller3",
//     assignedBot: null,
// },
// {
//     id: "3240112334234324324",
//     name: "controller4",
//     assignedBot: null,
// },
// ]);

export function assignJoystick(controllerId: string, botId: string) {
    controller.update((controllers) => {
    let controllerIndex

    if(controllerId === "") {
        controllerIndex = controllers.findIndex((c) => c.assignedBot === botId);
        controllers[controllerIndex].assignedBot = "";

    } else{
        controllerIndex = controllers.findIndex((c) => c.id === controllerId);
        // cheeck if bot is already assigned to another controller
        const alreadyAssigned = controllers.findIndex((c) => c.assignedBot === botId);
        if(alreadyAssigned !== -1) {
            controllers[alreadyAssigned].assignedBot = "";
        }
        controllers[controllerIndex].assignedBot = botId;
    }


    console.log("Assigned bot to controller", controllerId, botId);

    sendMessage({
        timestamp: new Date(),
        botId: botId,
        type: "joysticksChanged",
        data: {"joysticks":controllers},
    });
    return controllers;
    });
}