<script lang="ts">
	import Presentation from 'lucide-svelte/icons/presentation';
	import { Button } from '$lib/components/ui/button';
	import Logo from '$lib/components/custom/logo.svelte';
	import Map from '$lib/components/custom/map.svelte';

	import RobotList from '$lib/components/custom/robot-list.svelte';
	import Fullscreen from '$lib/components/custom/fullscreen.svelte';

	import { currentView } from '$lib/stores/main.js';

	import { botList } from '$lib/stores/stream.js';



	import {  activeBots } from '$lib/stores/main.js';
	import StreamStats from './stream-stats.svelte';

	function selectToggle() {
		if ($activeBots.length === Object.keys($botList).length) {
			$activeBots = [];
		} else {
			$activeBots = Object.keys($botList);
		}
	}


</script>

<div class="bg-muted border-r md:block relative">
	<div class="flex h-full max-h-screen flex-col gap-2">
		<div class="pl-auto flex h-14 flex-row p-4 pb-2 pt-6 lg:h-[60px]">
			<a href="/extensions/gui/_archive/nodejs_gui/frontend/static">
				<div class="h-full fill-neutral-400">
					<Logo />
				</div>
			</a>
			<Button
				class="ml-auto mr-2 h-8 w-8"
				variant="outline"
				size="icon"
				target="_blank"
				href="/present"><Presentation class="h-4 w-4" /></Button
			>
			<Fullscreen />
		</div>
		<div class="flex-1">
			<button
				class="w-full cursor-pointer p-4"
				on:click={(e) => {
					e.preventDefault();
					currentView.set('overview');
				}}
			>
				<div
					class="mb-2  w-full rounded-lg bg-white transition-[height]"
					class:h-0={$currentView == 'overview'}
					class:h-[15vh]={$currentView != 'overview'}
				>
					{#if $currentView != 'overview'}
						<Map mode="2D" positionDisplay = {false} overview={false} />
					{/if}
				</div>
				{#if $currentView != 'overview'}
					<Button
						class="w-full"
						on:click={(e) => {
							e.preventDefault();
							currentView.set('overview');
						}}>Overview</Button
					>
				{:else}
					<Button
						class="w-full"
						on:click={(e) => {
							e.preventDefault();
							selectToggle();
						}}
						>{$activeBots.length === Object.keys($botList).length
							? 'Deselect All'
							: 'Select All'}</Button
					>
				{/if}
			</button>

			<RobotList />
		</div>
	</div>
	<StreamStats />
</div>
