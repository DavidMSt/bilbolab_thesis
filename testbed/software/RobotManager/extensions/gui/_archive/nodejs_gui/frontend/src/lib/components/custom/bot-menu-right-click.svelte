<script lang="ts">
	import * as ContextMenu from '$lib/components/ui/context-menu';
	import { controller, assignJoystick } from '$lib/stores/controller';
	import Gamepad from 'lucide-svelte/icons/gamepad-2';
	import Power from 'lucide-svelte/icons/power';
	import { messages, getCurrentTimestamp, sendMessage } from '$lib/stores/messages'; // Import the messages and websocket stores
    import { onMount } from 'svelte';
    import { get } from 'svelte/store';

	export let bot;
    let currentController = $controller.find((con) => con?.assignedBot == bot?.id)?.id || 'none';
    let currentMode = 'off';
    let messageTemplates = [];

    onMount(() => {
        const loadedMessages = get(messages);
        messageTemplates = loadedMessages;
    });

    function assignController(value: string) {
        assignJoystick(value, bot.id);
    }

    function setControlMode(mode: string) {


        const message = { type: 'set', data : {}};
        message.botId = bot.id;
        message.data.key = `${bot.id}.control.mode`;
        message.data.value = mode;
        sendMessage(message);
        currentMode = mode;
    }
</script>

<ContextMenu.Content>
    <ContextMenu.Item on:click={() => setControlMode('off')}>
        <Power class="h-4" />Turn {bot.id} Off
    </ContextMenu.Item>

    <ContextMenu.Sub>
        <ContextMenu.SubTrigger>Control Mode</ContextMenu.SubTrigger>
        <ContextMenu.SubContent>
            <ContextMenu.RadioGroup bind:value={currentMode} onValueChange={setControlMode}>
                <ContextMenu.RadioItem  value={'off'} >
                    <div class="menu-item" >Off</div>
                </ContextMenu.RadioItem>
                <ContextMenu.RadioItem  value={'torque'}>
                    <div class="menu-item" >Torque</div>
                </ContextMenu.RadioItem>
                <ContextMenu.RadioItem  value={'balancing'}>
                    <div class="menu-item" >Balancing</div>
                </ContextMenu.RadioItem>
                <ContextMenu.RadioItem  value={'velocity'}>
                    <div class="menu-item" >Velocity</div>
            </ContextMenu.RadioItem>
        </ContextMenu.RadioGroup>
        </ContextMenu.SubContent>
    </ContextMenu.Sub>
    <ContextMenu.Separator />
    <ContextMenu.RadioGroup bind:value={currentController} onValueChange={assignController}>
        <ContextMenu.Label>Assign Controller</ContextMenu.Label>
        <ContextMenu.Separator />
        {#each $controller as con}
            <ContextMenu.RadioItem value={con.id}>
                <div class="inline-item mr-2 flex flex-row rounded-lg bg-neutral-300 p-0.5 pr-1 text-xs">
                    <Gamepad class="h-4 p-0" />{con.name}
                </div>
                {con.id}
            </ContextMenu.RadioItem>
        {/each}
        <ContextMenu.RadioItem value="">No Controller</ContextMenu.RadioItem>
    </ContextMenu.RadioGroup>
</ContextMenu.Content>

<style>
select, button {
    display: block;
    margin: 10px;
}
.selected {
        background-color: green;
        color: white;
}
</style>
