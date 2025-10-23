<script lang="ts">
	import uPlot, { type AlignedData, type TypedArray } from '$lib/components/custom/uplot.svelte';
	import { resize } from 'svelte-resize-observer-action';
	import 'uplot/dist/uPlot.min.css';
	import { onMount } from 'svelte';


	let plotContainer: HTMLDivElement;

	export const timeScroll: boolean = true;
	export const windowSize: number = 10;
	export const refreshInterval: number = 100;

	export let dataGetter: () => AlignedData; // Function that returns the data to plot

	let width: number;
	let height: number;

	let plot: uPlot;

    export let keys = ['time', 'v', 'psi_dot', 'psi', 'theta_dot', 'theta'];
	export let units = ['s', 'm/s', 'rad/s', 'deg','rad/s', 'deg'];
	const colors = [
		'black',
		'blue',
		'red',
		'green',
		'purple',
		'orange',
		'pink',
		'brown',
		'cyan',
		'magenta',
		'lightblue',
		'lightgreen',
		'lightyellow',
		'lightpurple',
		'lightorange',
		'lightpink',
		'lightbrown',
		'lightcyan',
		'lightmagenta'
	];
	export let ranges = [
		[null, null],
		[-3, 3],
		[-10, 10],
		[0, 360],
		[-10, 10],
		[-1.57, 1.57]
	];

	function onResize(entry: ResizeObserverEntry) {
		width = entry.contentRect.width;
		height = entry.contentRect.height - 25; // subtract the height of the legend
		plot.setSize({ width, height });
	}

	function initPlot() {
        let series = [];
		let axes = [];
		let scales = {
					x: {
						auto: false,
						time: true,
					}
				};

        for (let i = 0; i < keys.length; i++) {


            if (units[i] != 's') {
                series.push({
				label: keys[i],
				stroke: colors[i],
				width: 2,
				scale: keys[i]+units[i],
                axes: [units[i]],
				points: { show: false }
			});
				let side = i % 2 == 0 ? 1 : 3;
				axes.push({
					scale: keys[i]+units[i],
					label: keys[i] + ' [' + units[i] + ']',
					stroke: colors[i],
					side: side,
					size: 40
				});
				let key = units[i].replace('/', '_');
				//check if the key is already in the scales
				if (!((keys[i]+units[i]) in scales)) {
					const rangePadding= (ranges[i][1]-ranges[i][0])*0.05;
					scales[keys[i]+units[i]] = { auto: false, range: [ranges[i][0]-rangePadding, ranges[i][1]+rangePadding] };
				}
			} else {
                series.push({});
                axes.push({});

			}
        }


		const opts = {
			width: width,
			height: height,
			series: series,
			scales: scales,
			axes: axes
		};

		plot = new uPlot(opts, [], plotContainer);
	}

	function scroll() {
		if (timeScroll === false) {
			setTimeout(scroll, 100);
			return;
		}
		const currentTime = Date.now() / 1000;
		plot.setScale('x', {
			min: currentTime - windowSize, // Show last 10 seconds
			max: currentTime // start drawing off screen
		});
		requestAnimationFrame(scroll);
	}

	function updateData() {
        const data = dataGetter();
        if (data?.length && data[0]?.length && data[0][1]?.length){
            const currentTime = Date.now() / 1000;
		    plot.setData(data[0], false);
        }
	}

	onMount(async () => {
		initPlot();
		requestAnimationFrame(scroll);
		if (refreshInterval > 0) setInterval(updateData, refreshInterval);
	});
</script>

<div class="h-full w-full p-2">
	<div class="h-full w-full" bind:this={plotContainer} use:resize={onResize}></div>
</div>

<style>
	:global(.u-legend) {
		height: 28px !important;
		overflow: hidden !important;
		font-size: 14px !important;
	}

</style>
