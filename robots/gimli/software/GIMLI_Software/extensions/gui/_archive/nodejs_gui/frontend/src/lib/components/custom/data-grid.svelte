<script lang="ts">
    import Grid, {GridItem, type GridController} from 'svelte-grid-extended';
    import {Button} from '$lib/components/ui/button';

    import CircleX from 'lucide-svelte/icons/circle-x';
    import Pencil from 'lucide-svelte/icons/pencil';
    import Map from '$lib/components/custom/map.svelte';
    import * as Popover from '$lib/components/ui/popover';
    import {Label} from '$lib/components/ui/label';
    import {Input} from '$lib/components/ui/input';
    import * as Select from '$lib/components/ui/select';
    import RobotList from '$lib/components/custom/robot-list.svelte';
    import Terminal from '$lib/components/custom/terminal.svelte';

    import Plot from '$lib/components/custom/plot.svelte';

    export let gridId: string;
    export let presentationMode = false;
    export let fill = false;

    export let defaultLayout = [{
        id: Date.now().toString(),
        x: 0,
        y: 0,
        w: 4,
        h: 2,
        type: 'map',
        title: '',
        content: {}
    }];

    let items = localStorage.getItem(('grid:' + gridId)) ? JSON.parse(localStorage.getItem(('grid:' + gridId))) : defaultLayout;
    const itemSize = {height: 100};

    export let locked = true;

    let gridController: GridController;

    function addItem() {
        const newPosition = gridController.getFirstAvailablePosition(4, 1);

        items = newPosition
            ? [
                ...items,
                {
                    id: Date.now().toString(),
                    x: newPosition.x,
                    y: newPosition.y,
                    w: 4,
                    h: 2,
                    title: '',
                    type: '',
                    content: {}
                }
            ]
            : items;
    }

    function remove(id: string) {
        items = items.filter((i) => i.id !== id);
    }

    $: items, localStorage.setItem('grid:' + gridId, JSON.stringify(items));

</script>

{#if items.length !== 0}
    <Grid
            itemSize={fill ? undefined : itemSize}
            gap={10}
            cols={4}
            rows={fill ? 8 : undefined}
            bounds
            collision={fill ? undefined : "compress"}
            class="p-0 overflow-hidden"
            bind:controller={gridController}
            readOnly={locked}
    >
        {#each items as item}
            <GridItem
                    bind:x={item.x}
                    bind:y={item.y}
                    bind:w={item.w}
                    bind:h={item.h}
                    class="overflow-hidden rounded-lg border"
                    activeClass="shadow-lg bg-neutral-100 opacity-90"
                    previewClass="bg-neutral-400 rounded-lg transition-all"
            >
                {#if !locked}
                    <div class="absolute right-0 top-0 z-20 flex gap-4 p-2 text-xl">
                        <Popover.Root bind:open={item.popoverOpen}>
                            <Popover.Trigger
                                    class="!focus:outline-none !focus:border-transparent !focus:ring-0 !border-transparent !outline-none"
                            >
                                <button
                                        on:pointerdown={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                    }}
                                        class="hover:text-primary h-fit cursor-pointer rounded-full text-neutral-500 transition-all hover:scale-125 hover:bg-neutral-300"
                                        on:click={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        item.popoverOpen = true;
                    }}
                                >
                                    <Pencil class="text-xl"/>
                                </button>
                            </Popover.Trigger>
                            <Popover.Content class="w-80">
                                <div class="grid gap-4">
                                    <div class="grid gap-2">
                                        <div class="grid grid-cols-3 items-center gap-4">
                                            <Label for="width">Block Type</Label>
                                            <Select.Root
                                                    selected={{ value: item.type, label: item.type }}
                                                    onSelectedChange={(v) => {
                                    v && (item.type = v.value);
                                    if (v.value !== 'map') {
                                        item.mapMode = null;
                                    }
                                }}
                                            >
                                                <Select.Trigger class="w-[180px]">
                                                    <Select.Value placeholder="Block Type"/>
                                                </Select.Trigger>
                                                <Select.Content>
                                                    <Select.Item value="map">Map</Select.Item>
                                                    <Select.Item value="timeseries">Timeseries</Select.Item>
                                                    <Select.Item value="terminal">Terminal</Select.Item>
                                                    <Select.Item value="robotlist">Robot List</Select.Item>
                                                </Select.Content>
                                            </Select.Root>
                                        </div>
                                        {#if item.type === 'map'}
                                            <div class="grid grid-cols-3 items-center gap-4">
                                                <Label for="mapMode">Map Mode</Label>
                                                <Select.Root
                                                        selected={{ value: item.mapMode, label: item.mapMode }}
                                                        onSelectedChange={(v) => {
                                        v && (item.mapMode = v.value);
                                    }}
                                                >
                                                    <Select.Trigger class="w-[180px]">
                                                        <Select.Value placeholder="Map Mode"/>
                                                    </Select.Trigger>
                                                    <Select.Content>
                                                        <Select.Item value="2D">2D</Select.Item>
                                                        <Select.Item value="3D">3D</Select.Item>
                                                        <Select.Item value="Toggle">Toggle</Select.Item>
                                                    </Select.Content>
                                                </Select.Root>
                                            </div>
                                        {:else if item.type === 'timeseries'}
                                            <div class="grid grid-cols-3 items-center gap-4">
                                                <Label for="BotID">BotID</Label>
                                                <Select.Root
                                                        selected={{ value: item.BotID, label: item.BotID }}
                                                        onSelectedChange={(v) => {
                                        v && (item.BotID = v.value);
                                    }}
                                                >
                                                    <Select.Trigger class="w-[180px]">
                                                        <Select.Value placeholder="BotID"/>
                                                    </Select.Trigger>
                                                    <Select.Content>
                                                        <Select.Item value="0">0</Select.Item>
                                                        <Select.Item value="1">1</Select.Item>
                                                        <Select.Item value="2">2</Select.Item>
                                                        <Select.Item value="3">3</Select.Item>
                                                        <Select.Item value="4">4</Select.Item>
                                                        <Select.Item value="5">5</Select.Item>
                                                        <Select.Item value="6">6</Select.Item>
                                                        <Select.Item value="7">7</Select.Item>
                                                        <Select.Item value="8">8</Select.Item>
                                                        <Select.Item value="9">9</Select.Item>
                                                    </Select.Content>
                                                </Select.Root>
                                            </div>
                                            <div class="grid grid-cols-3 items-center gap-4">
                                                <Label for="Data">Data</Label>
                                                <Select.Root
                                                        selected={{ value: item.data_plot, label: item.data_plot }}
                                                        onSelectedChange={(v) => {
                                        v && (item.data_plot = v.value);
                                    }}
                                                >
                                                    <Select.Trigger class="w-[180px]">
                                                        <Select.Value placeholder="Data"/>
                                                    </Select.Trigger>
                                                    <Select.Content>
                                                        <Select.Item value="Psi">Psi</Select.Item>
                                                        <Select.Item value="Psi_dot">Psi_dot</Select.Item>
                                                        <Select.Item value="Theta">Theta</Select.Item>
                                                        <Select.Item value="Theta_dot">Theta_dot</Select.Item>
                                                        <Select.Item value="v">v</Select.Item>
                                                    </Select.Content>
                                                </Select.Root>
                                            </div>
                                        {/if}
                                        <div class="grid grid-cols-3 items-center gap-4">
                                            <Label for="maxWidth">Title</Label>
                                            <Input id="maxWidth" bind:value={item.title} class="col-span-2 h-8"/>
                                        </div>
                                    </div>
                                </div>
                            </Popover.Content>
                        </Popover.Root>
                        <button
                                class="hover:text-destructive h-fit rounded-full text-neutral-500 transition-all hover:scale-125 hover:bg-neutral-300"
                                on:pointerdown={(e) => e.stopPropagation()}
                                on:click={() => remove(item.id)}
                        >
                            <CircleX/>
                        </button>
                    </div>
                {/if}
                <div class=" h-full w-full ">
                    {#if item.title?.length > 0}
                        <div class=" absolute z-10 top-0 left-0 flex p-1 w-full items-center justify-center font-semibold">
                            <span class="bg-white rounded-full px-2 ">{item.title}</span>
                        </div>
                    {/if}
                    {#if item.type === 'map' && item.mapMode === '2D'}
                        <Map presentation={presentationMode} mode={'2D'}/>
                    {:else if item.type === 'map' && item.mapMode === '3D'}
                        <Map presentation={presentationMode} mode={'3D'}/>
                    {:else if item.type === 'map' && item.mapMode === 'Toggle'}
                        <Map presentation={presentationMode} mode={'toggle'}/>
                    {:else if item.type === 'timeseries'}
                        <Plot/>
                    {:else if item.type === 'robotlist'}
                        <RobotList showActive={false}/>
                    {:else if item.type === 'terminal'}
                        <Terminal showInput={false}/>
                    {:else}
                        <div class="bg-muted flex h-full items-center justify-center font-semibold">
                            Select Datafield and Blocktype
                        </div>
                    {/if}

                </div>
            </GridItem>
        {/each}
    </Grid>
{/if}
{#if !locked}
    <div class="ml-[10px] py-5 mb-12 bottom-2 right-10" class:absolute={fill}>
        <Button
                class=" bg-muted text-primary h-20 w-full text-lg hover:bg-neutral-300"
                on:click={addItem}>+ Add Item
        </Button
        >
    </div>
{/if}
