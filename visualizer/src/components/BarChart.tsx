/**
 * D3-based bar chart with error bars for displaying benchmark results.
 */

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface BarChartData {
  method: string;
  accuracy: number;
  n_correct: number;
  n_total: number;
  ci_lower: number;
  ci_upper: number;
  rank: number;
  isSelected: boolean;
  isAggregate?: boolean;
}

interface BarChartProps {
  data: BarChartData[];
  onMethodSelect: (method: string) => void;
  width?: number;
  height?: number;
}

export const BarChart: React.FC<BarChartProps> = ({
  data,
  onMethodSelect,
  width = 600,
  height = 400,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Set up dimensions
    const margin = { top: 20, right: 30, bottom: 60, left: 120 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create scales
    const xScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([0, chartWidth]);

    const yScale = d3
      .scaleBand()
      .domain(data.map((d) => d.method))
      .range([0, chartHeight])
      .padding(0.2);

    // Color scale for medals
    const getBarColor = (rank: number) => {
      if (rank === 1) return '#FFD700'; // Gold
      if (rank === 2) return '#C0C0C0'; // Silver
      if (rank === 3) return '#CD7F32'; // Bronze
      return '#4a90e2'; // Blue
    };

    const getBarColorDark = (rank: number) => {
      if (rank === 1) return '#FFA500';
      if (rank === 2) return '#A0A0A0';
      if (rank === 3) return '#8B4513';
      return '#2171d6';
    };

    // Add X axis
    const xAxis = g
      .append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(
        d3
          .axisBottom(xScale)
          .ticks(5)
          .tickFormat((d) => `${(d.valueOf() * 100).toFixed(0)}%`)
      );

    xAxis.selectAll('text').style('font-size', '12px');
    xAxis.selectAll('line').style('stroke', '#ccc');
    xAxis.select('.domain').style('stroke', '#ccc');

    // Add Y axis
    const yAxis = g.append('g').call(d3.axisLeft(yScale));

    yAxis.selectAll('text').style('font-size', '12px').style('font-weight', '500');
    yAxis.selectAll('line').style('stroke', '#ccc');
    yAxis.select('.domain').style('stroke', '#ccc');

    // Add gridlines
    g.append('g')
      .attr('class', 'grid')
      .call(
        d3
          .axisBottom(xScale)
          .ticks(5)
          .tickSize(chartHeight)
          .tickFormat(() => '')
      )
      .call((g) => g.select('.domain').remove())
      .call((g) =>
        g.selectAll('.tick line').attr('stroke', '#f0f0f0').attr('stroke-opacity', 0.7)
      );

    // Add error bars (confidence intervals)
    g.selectAll('.error-bar')
      .data(data)
      .enter()
      .append('line')
      .attr('class', 'error-bar')
      .attr('x1', (d) => xScale(d.ci_lower))
      .attr('x2', (d) => xScale(d.ci_upper))
      .attr('y1', (d) => yScale(d.method)! + yScale.bandwidth() / 2)
      .attr('y2', (d) => yScale(d.method)! + yScale.bandwidth() / 2)
      .attr('stroke', (d) => getBarColorDark(d.rank))
      .attr('stroke-width', 2)
      .attr('opacity', 0.5);

    // Add error bar caps
    g.selectAll('.error-cap-left')
      .data(data)
      .enter()
      .append('line')
      .attr('class', 'error-cap-left')
      .attr('x1', (d) => xScale(d.ci_lower))
      .attr('x2', (d) => xScale(d.ci_lower))
      .attr('y1', (d) => yScale(d.method)! + yScale.bandwidth() / 2 - 5)
      .attr('y2', (d) => yScale(d.method)! + yScale.bandwidth() / 2 + 5)
      .attr('stroke', (d) => getBarColorDark(d.rank))
      .attr('stroke-width', 2)
      .attr('opacity', 0.5);

    g.selectAll('.error-cap-right')
      .data(data)
      .enter()
      .append('line')
      .attr('class', 'error-cap-right')
      .attr('x1', (d) => xScale(d.ci_upper))
      .attr('x2', (d) => xScale(d.ci_upper))
      .attr('y1', (d) => yScale(d.method)! + yScale.bandwidth() / 2 - 5)
      .attr('y2', (d) => yScale(d.method)! + yScale.bandwidth() / 2 + 5)
      .attr('stroke', (d) => getBarColorDark(d.rank))
      .attr('stroke-width', 2)
      .attr('opacity', 0.5);

    // Add bars
    const bars = g
      .selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', 0)
      .attr('y', (d) => yScale(d.method)!)
      .attr('width', (d) => xScale(d.accuracy))
      .attr('height', yScale.bandwidth())
      .attr('fill', (d) => getBarColor(d.rank))
      .attr('stroke', (d) => (d.isSelected ? '#000' : 'none'))
      .attr('stroke-width', (d) => (d.isSelected ? 2 : 0))
      .attr('opacity', (d) => (d.isAggregate ? 0.7 : 0.9))
      .style('cursor', (d) => (d.isAggregate ? 'default' : 'pointer'))
      .on('mouseover', function (event, d) {
        if (!d.isAggregate) {
          d3.select(this).attr('opacity', 1);
        }
      })
      .on('mouseout', function (event, d) {
        d3.select(this).attr('opacity', d.isAggregate ? 0.7 : 0.9);
      })
      .on('click', (event, d) => {
        if (!d.isAggregate) {
          onMethodSelect(d.method);
        }
      });

    // Add tooltips
    bars.append('title').text(
      (d) =>
        d.isAggregate
          ? `${d.method}\n` +
            `Mean Accuracy: ${(d.accuracy * 100).toFixed(1)}%\n` +
            `Total Correct: ${d.n_correct}/${d.n_total}\n` +
            `95% CI: [${(d.ci_lower * 100).toFixed(1)}%, ${(d.ci_upper * 100).toFixed(1)}%]\n\n` +
            `(Aggregate rows have no individual embeddings)`
          : `${d.method}\n` +
            `Accuracy: ${(d.accuracy * 100).toFixed(1)}%\n` +
            `Correct: ${d.n_correct}/${d.n_total}\n` +
            `95% CI: [${(d.ci_lower * 100).toFixed(1)}%, ${(d.ci_upper * 100).toFixed(1)}%]`
    );

    // Add value labels
    g.selectAll('.label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', (d) => xScale(d.accuracy) + 5)
      .attr('y', (d) => yScale(d.method)! + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .text((d) => `${(d.accuracy * 100).toFixed(1)}%`)
      .style('font-size', '11px')
      .style('font-weight', '600')
      .style('fill', '#333');

    // Add rank medals to Y axis labels
    yAxis.selectAll('.tick text').each(function (d, i) {
      const rank = data[i].rank;
      let medal = '';
      if (rank === 1) medal = '🥇 ';
      if (rank === 2) medal = '🥈 ';
      if (rank === 3) medal = '🥉 ';
      d3.select(this).text(medal + d);
    });
  }, [data, onMethodSelect, width, height]);

  return (
    <div style={{ background: '#fff', padding: '20px', borderRadius: '4px' }}>
      <svg ref={svgRef}></svg>
      <div
        style={{
          marginTop: '12px',
          fontSize: '11px',
          color: '#888',
          textAlign: 'center',
        }}
      >
        Error bars show 95% confidence intervals. Click bars to select method.
      </div>
    </div>
  );
};
