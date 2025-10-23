<script>
	import { Label } from 'bits-ui';
	import { onMount, onDestroy } from 'svelte';
	import { scale } from 'svelte/transition';
	import uPlot from '$lib/components/custom/uplot.svelte';
	import 'uplot/dist/uPlot.min.css';
	//import { i } from 'vitest/dist/reporters-yx5ZTtEV.js';

	let data = [];
	let chart;
	let chartElement;
	let updateInterval;
	let currentTime;
	export let keys = ['time', 'v', 'psi_dot', 'psi'];
	export let units = ['s', 'm/s', 'rad/s', 'deg'];
	export let plotData = [[], [], [], []];
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
		[0, 1],
		[-1, 1],
		[0, 360]
	];
	let width;
	let height;
	let legend_height = 50;

	onMount(() => {
		//console.log('Mounted RobotPlot');
		window.addEventListener('resize', resizePlot);
		initPlot();
		startDataUpdate();
	});

	function resizePlot() {
		if (chart) {
			chart.setSize({ width: width, height: height - legend_height * 2 });
		}
	}

	function getLegendHeight() {
		requestAnimationFrame(() => {
			const legend = document.querySelector('.u-legend');
			if (legend) {
				const lheight = legend.getBoundingClientRect().height;
				legend_height = lheight;
			} else {
				console.log('Legend not found');
			}
		});
	}
	// execute resizePlot when width or height changes
	$: if (width && height) {
		getLegendHeight();
		resizePlot();
	}

	function initPlot() {
		const containerWidth = chartElement.offsetWidth;
		let series = [];
		let axes = [];
		let scales = { x: { time: true } };

		for (let i = 0; i < keys.length; i++) {
			series.push({
				label: keys[i],
				stroke: colors[i],
				width: 2,
				scale: units[i],
				points: { show: false }
			});
			if (units[i] != 's') {
				let side = i % 2 == 0 ? 1 : 3;
				axes.push({
					scale: units[i],
					values: (u, vals) => vals.map((v) => v.toFixed(0)),
					label: keys[i] + ' [' + units[i] + ']',
					stroke: colors[i],
					side: side,
					size: 40
				});
				let key = units[i].replace('/', '_');
				// check if the key is already in the scales
				if (!(key in scales)) {
					scales[units[i].replace('/', '_')] = { range: ranges[i] };
				}
			} else {
				axes.push({
					scale: units[i].replace('/', '_'),
					values: (u, vals) => vals.map((v) => v.toFixed(2)),
					label: keys[i] + ' [' + units[i] + ']',
					stroke: colors[i]
				});
			}
		}

		const opts = {
			width: width,
			height: height * 0.8,
			title: 'Live data',
			series: series,
			scales: scales,
			axes: axes
		};

		chart = new uPlot(opts, data, chartElement);
	}

	function update() {
		const currentTime = Date.now() / 1000;

		chart.setScale('x', {
			min: currentTime - 10, // Show last 10 seconds
			max: currentTime - 2 // start drawing off screen
		});

		requestAnimationFrame(update);
	}

	function startDataUpdate() {
		updateInterval = setInterval(() => {
			// falls ihr das mit relativen Zeiten haben wollt, dann hier das benutzen (man muss dafür dann die skala auch ein bisschen anpassen)
			// data = [...plotData];
			// data[0] = data[0].map((time) => (time * 1000 - startTime) / 100);

			chart.setData(plotData);
		}, 100); // Update data every second
		requestAnimationFrame(update);
	}

	onDestroy(() => {
		window.removeEventListener('resize', resizePlot);
		clearInterval(updateInterval);
		if (chart) chart.destroy();
	});
</script>

<div
	class="h-full w-full"
	bind:this={chartElement}
	bind:clientHeight={height}
	bind:clientWidth={width}
></div>

<style>
	div {
		width: 100%; /* Full width of its parent */
		overflow-x: auto; /* Allows horizontal scrolling */
	}
</style>
