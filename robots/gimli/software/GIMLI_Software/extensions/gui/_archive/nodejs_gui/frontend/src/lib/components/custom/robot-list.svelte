<script lang="ts">
    import {Badge} from '$lib/components/ui/badge';
    import {Separator} from '$lib/components/ui/separator';
    import Battery from '$lib/components/custom/battery.svelte';

    import {currentBot, currentView, activeBots} from '$lib/stores/main.js';
    import {botColor} from '$lib/helpers/bot-colors';
    import {shortcut} from '$lib/helpers/shortcut';
    import Gamepad from 'lucide-svelte/icons/gamepad-2';

    import * as ContextMenu from '$lib/components/ui/context-menu';

    import {botList} from '$lib/stores/stream';
    import BotMenuRightClick from '$lib/components/custom/bot-menu-right-click.svelte';
    import {controller} from '$lib/stores/controller';

    export let showActive = true;

    function setCurrentBot(i: string) {
        if ($currentView === 'detail') {
            currentBot.set(i);
        } else {
            if ($activeBots.includes(i)) {
                $activeBots = $activeBots.filter((bot) => bot != i);
            } else {
                $activeBots = [...$activeBots, i];
            }
        }
    }

    $: isActive = (i) => {
        if (showActive) {
            if ($currentView === 'detail') {
                return $currentBot === i;
            } else {
                return $activeBots.includes(i);
            }
        } else {
            return true;
        }
    };
</script>

<Separator/>
<nav class=" w-full overflow-hidden text-sm font-medium">
    {#if Object.keys($botList).length === 0}
        <div class="text-muted-foreground flex h-[7vh] items-center px-4">No bots connected</div>
        <Separator/>
    {/if}
    {#each Object.entries($botList).sort((a, b) => a[1].number - b[1].number) as [i, bot]}
        <ContextMenu.Root>
            <ContextMenu.Trigger>
                <a
                        href="##"
                        on:click={(e) => {
						e.preventDefault();
						setCurrentBot(bot.id);
					}}
                        on:dblclick={(e) => {
						e.preventDefault();
						setCurrentBot(bot.id);
						currentView.set('detail');
					}}
                        use:shortcut={{
						control: true,
						code: 'Digit' + bot.number,
						callback: () => setCurrentBot(bot.id)
					}}
                        class="text-muted-foreground hover:text-primary vertical flex h-[7vh] w-full flex-row items-center px-4 transition-all duration-100"
                        class:bg-background={isActive(bot.id)}
                        class:font-bold={isActive(bot.id)}
                        class:text-primary={isActive(bot.id)}
                        class:opacity-70={!isActive(bot.id)}
                >
                    <div
                            class="bg-muted mr-2 flex h-6 w-6 flex-none items-center justify-center rounded-full text-xs text-white"
                            style="background-color: {botColor(bot.number)};"
                    >
                        {bot.number}
                    </div>
                    <div class="shrink truncate">
                        <h3>{bot.id}</h3>
                        <p class=" w-full truncate text-xs text-gray-400">
                            {bot.status} | {bot.controlMode} | {bot.estimationMode}
                        </p>
                    </div>
                    <div class="flex flex-none grow flex-col pl-2">
                        <div class=" mx-auto mb-1 h-4 w-4">
                            <Battery voltage={bot.battery} charging={bot.charging}/>
                        </div>
                        {#if $controller.find((con) => con?.assignedBot == bot?.id)}
                            <div class="mx-auto flex w-fit flex-row rounded-lg bg-neutral-300 p-0.5 pr-2 text-xs">
                                <Gamepad class="h-4 p-0"/>
                                {$controller.find((con) => con?.assignedBot == bot?.id)?.name}
                            </div>
                        {/if}
                    </div>
                </a>
            </ContextMenu.Trigger>
            <BotMenuRightClick {bot}/>
        </ContextMenu.Root>
        <Separator/>
    {/each}
</nav>
