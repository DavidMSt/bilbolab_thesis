<script>
	import { stream } from '$lib/stores/stream.js';
	import { currentBot } from '$lib/stores/main.js';
	import JsonSheet from '$lib/components/custom/json-sheet.svelte';
	import * as Resizable from '$lib/components/ui/resizable/index.js';
	import { botColor } from '$lib/helpers/bot-colors.js';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import Parameters from '$lib/components/custom/parameters.svelte';
	import DataGrid from '$lib/components/custom/data-grid.svelte';
	import Button from '../ui/button/button.svelte';
	import Pencil from 'lucide-svelte/icons/pencil';
	import Lock from 'lucide-svelte/icons/lock';
	let locked = true;
</script>

<Resizable.PaneGroup direction="horizontal" autoSaveId="pane-oberview-h">
	<Resizable.Pane>
		<div class="h-full overflow-scroll p-10 pb-10">
			<div class="flex items-center justify-between p-[10px]">

				<h1 class="text-lg font-semibold md:text-2xl"><div class="mb-4 inline-block text-2xl font-semibold">
					<Badge
					class="bg-muted mr-2 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-lg"
					style="background-color: {botColor($currentBot)};">
					{parseInt($currentBot.replace(/\D/g, ''))}
					</Badge>
				</div> Robot {$currentBot}</h1>
				<Button on:click={() => (locked = !locked)} class="bg-neutral-400">
					{#if !locked}
						<Lock />
					{:else}
						<Pencil />
					{/if}
				</Button>
			</div>
			<div class="w-full ">
				<DataGrid {locked} gridId={"bot-details"} />
			</div>
		</div>
	</Resizable.Pane>
	<Resizable.Handle withHandle />
	<Resizable.Pane defaultSize={25}  class="bg-muted p-4"><Parameters/></Resizable.Pane>
</Resizable.PaneGroup>
