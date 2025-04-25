import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import Chart from 'chart.js/auto';
import { LDAVisualizationsProps } from '@/app/types/pipeline';

const getColorForAttribute = (attribute: string) => {
  const colorMap: { [key: string]: string } = {
    male: 'rgba(54, 162, 235, 0.6)',
    female: 'rgba(255, 99, 132, 0.6)',
  };
  if (colorMap[attribute]) return colorMap[attribute];
  const r = Math.floor(Math.random() * 200 + 55);
  const g = Math.floor(Math.random() * 200 + 55);
  const b = Math.floor(Math.random() * 200 + 55);
  return `rgba(${r}, ${g}, ${b}, 0.6)`;
};

export function LDAVisualizations({ ldaResults }: LDAVisualizationsProps) {
  const overallChartRef = useRef<HTMLCanvasElement>(null);
  const demographicChartRef = useRef<HTMLCanvasElement>(null);

  // Compute overall average topic probabilities across all documents.
  const overallAverages = useMemo(() => {
    if (!ldaResults.distributions || ldaResults.distributions.length === 0) return [];
    const nTopics = ldaResults.distributions[0].length;
    const sums = Array(nTopics).fill(0);
    ldaResults.distributions.forEach(dist => {
      dist.forEach((value, idx) => {
        sums[idx] += value;
      });
    });
    return sums.map(sum => sum / ldaResults.distributions.length);
  }, [ldaResults.distributions]);

  // Get demographic categories from the LDA 
  const demographicCategories = useMemo(() => {
    if (!ldaResults.demographicDistributions) return [];
    return Object.keys(ldaResults.demographicDistributions);
  }, [ldaResults.demographicDistributions]);

  const [selectedCategory, setSelectedCategory] = useState<string>('');

  // Ensure selected category is valid 
  useEffect(() => {
    if (demographicCategories.length > 0) {
      setSelectedCategory(prevCategory =>
        demographicCategories.includes(prevCategory) ? prevCategory : demographicCategories[0]
      );
    }
  }, [demographicCategories]);

  useEffect(() => {
  }, [selectedCategory]);

  useEffect(() => {
    // Clean up existing charts before rendering new ones
    [overallChartRef, demographicChartRef].forEach(ref => {
      if (ref.current) {
        const existingChart = Chart.getChart(ref.current);
        if (existingChart) {
          existingChart.destroy();
        }
      }
    });

    // --- Overall Average Topic Distribution Chart ---
    const overallCtx = overallChartRef.current?.getContext('2d');
    if (overallCtx && overallAverages.length > 0) {
      const labels = ldaResults.topics.map((_, idx) => `Topic ${idx}`);
      new Chart(overallCtx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Average Topic Probability',
            data: overallAverages,
            backgroundColor: 'rgba(75, 192, 192, 0.6)'
          }]
        },
        options: {
          responsive: true,
          plugins: { 
            title: { display: true, text: 'Overall Average Topic Distribution' } 
          },
          scales: {
            x: { beginAtZero: true },
            y: { beginAtZero: true }
          }
        }
      });
    }

    // --- Demographic Distribution Chart ---
    const demographicCtx = demographicChartRef.current?.getContext('2d');
    if (demographicCtx && selectedCategory && ldaResults.demographicDistributions) {
      let subgroupData = ldaResults.demographicDistributions[selectedCategory];

if (selectedCategory !== "baseline" && ldaResults.demographicDistributions["baseline"]) {
  subgroupData = {
    ...ldaResults.demographicDistributions["baseline"],
    ...subgroupData
  };
}

if (!subgroupData || typeof subgroupData !== "object") {
  console.error("Invalid subgroupData for:", selectedCategory, subgroupData);
  return;
}

const labels = ldaResults.topics.map((_, idx) => `Topic ${idx}`);
const datasets = Object.entries(subgroupData).map(([subgroup, avgVector]) => ({
  label: subgroup,
  data: Array.isArray(avgVector) ? avgVector : [],
  backgroundColor: getColorForAttribute(subgroup)
}));

if (datasets.length === 0) {
  console.warn("No datasets available for demographic chart");
  return;
}


      new Chart(demographicCtx, {
        type: 'bar',
        data: {
          labels,
          datasets
        },
        options: {
          responsive: true,
          plugins: { 
            title: { display: true, text: `Average Topic Distribution by ${selectedCategory}` } 
          },
          scales: {
            y: { beginAtZero: true, title: { display: true, text: 'Average Probability' } }
          }
        }
      });
    }
  }, [ldaResults, overallAverages, selectedCategory]);



  return (
    <div className="grid grid-cols-1 gap-6">

      <Card>
        <CardContent className="pt-4">
          <h3 className="text-lg font-semibold mb-4">Overall Average Topic Distribution</h3>
          <canvas ref={overallChartRef} />
        </CardContent>
      </Card>

      {demographicCategories.length > 0 && (
        <div className="flex items-center gap-2">
          <label htmlFor="demographic-select" className="font-medium">
            Filter by Demographic Category:
          </label>
          <select
            id="demographic-select"
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="border rounded p-1"
          >
            {demographicCategories.map(category => (
              <option key={category} value={category}>{category}</option>
            ))}
          </select>
        </div>
      )}

      <Card>
        <CardContent className="pt-4">
          <h3 className="text-lg font-semibold mb-4">
            Average Topic Distribution by {selectedCategory}
          </h3>
          <canvas ref={demographicChartRef} />
        </CardContent>
      </Card>
    </div>
  );
}
