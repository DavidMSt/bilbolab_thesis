import { writable } from 'svelte/store';
import { controller } from './controller';

interface Message {
    timestamp: string;
    botId: string | null;
    type: string;
    data: {
        key?: string;
        value?: string;
        command?: string;
        args?: string;
        assignedBot?: string;
        botId?: string;
        controllerId?: string ;
    }

}


export const messages = writable<Message[]>([]);
let ws: WebSocket;

export function sendMessage(message: Message) {
    const m = message;
    m.timestamp= new Date();

    messages.update((messages) => {
        messages.push(m);
        return messages;
    });
    ws.send(JSON.stringify(m));
}

function messageHandler(message: Message) {

    messages.update((m) => {
        m.push(message);
        return m;
    });

    if (message.type === "joysticksChanged") {
        controller.set(message.data.joysticks);
    }
}


export function initializeWebSocket() {
    ws = new WebSocket("ws://localhost:8766");

    ws.addEventListener("open", function (event) {
        console.log("Connected to WS Server");
    });

    ws.addEventListener("message", function (event) {
        console.log('Message from server ', event.data);
        messageHandler(JSON.parse(event.data));

    });

    ws.addEventListener("error", function (event) {
        console.error("WebSocket error:", event);
        messages.update((messages) => {
            messages.push(event.data);
            return messages;
        });
    });


}

export function getCurrentTimestamp(): string {
    return new Date();
}



// check if websocket is open otherwise reconnect
setInterval(() => {
    if (ws.readyState !== ws.OPEN) {
        console.log("Attempting reconnect to WS Server");
        ws.close();
        initializeWebSocket();
    }
},500);
