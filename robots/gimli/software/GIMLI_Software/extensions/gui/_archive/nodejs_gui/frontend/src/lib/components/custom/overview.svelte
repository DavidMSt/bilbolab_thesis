<script>
	import Map from '$lib/components/custom/map.svelte';
	import Button from '../ui/button/button.svelte';
	import Pencil from 'lucide-svelte/icons/pencil';
	import Lock from 'lucide-svelte/icons/lock';

	let locked = true;
	import { activeBots } from '$lib/stores/main.js';
	import DataGrid from '$lib/components/custom/data-grid.svelte';
	let lat;
	let freq;

</script>

<div class="h-full p-10 pb-10 overflow-scroll">
	<div class="flex items-center justify-between p-[10px]">
		<h1 class="mb-4 text-2xl font-semibold">Overview</h1>
		<Button on:click={() => (locked = !locked)} class="bg-neutral-400">
			{#if !locked}
				<Lock />
			{:else}
				<Pencil />
			{/if}
		</Button>
	</div>
	<div class="w-full">
		<DataGrid {locked} gridId={"overview"} />
	</div>
	<!-- <div>
		Show: {$activeBots}<br>
	Latency: {lat} s<br>
	Frequency: {freq} Hz
	</div> -->
</div>
