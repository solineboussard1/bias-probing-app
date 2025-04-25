import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Chart from 'chart.js/auto';
import { DownloadIcon } from "lucide-react";
import { createConceptExtractionCSV, downloadCSV } from "@/app/lib/csv-utils";
import {ClusterData, ConceptVisualizationsProps } from "@/app/types/pipeline";

export type ClusterOutput = {
  all: ClusterData[];
  demographics: { [key: string]: ClusterData[] };
};

const colorMap: { [key: string]: string } = {
  male: 'rgba(54, 162, 235, 0.6)',
  female: 'rgba(255, 99, 132, 0.6)',
};

const getColorForAttribute = (attribute: string) => {
  if (colorMap[attribute]) return colorMap[attribute];
  const r = Math.floor(Math.random() * 200 + 55);
  const g = Math.floor(Math.random() * 200 + 55);
  const b = Math.floor(Math.random() * 200 + 55);
  return `rgba(${r}, ${g}, ${b}, 0.6)`;
};

export function ConceptVisualizations({ conceptData }: ConceptVisualizationsProps) {
  // Two chart refs.
  const overallChartRef = useRef<HTMLCanvasElement>(null);
  const distributionChartRef = useRef<HTMLCanvasElement>(null);

  const clustersData = useMemo<ClusterData[]>(() => {
    return conceptData.clusters?.all ?? [];
  }, [conceptData.clusters]);

  const demographicCategories = useMemo(() => {
    if (!conceptData.clusters?.demographics) return [];
    const keys = Object.keys(conceptData.clusters.demographics);
    const groups = keys.map(key => key === "baseline" ? "baseline" : key.split(':')[0]);
    return Array.from(new Set(groups));
  }, [conceptData.clusters?.demographics]);
  
  // Initialize selectedGroup from unique categories.
  const [selectedGroup, setSelectedGroup] = useState<string>(demographicCategories[0] || "");
  
  useEffect(() => {
    if (demographicCategories.length > 0 && !demographicCategories.includes(selectedGroup)) {
      setSelectedGroup(demographicCategories[0]);
    }
  }, [demographicCategories, selectedGroup]);

  useEffect(() => {
    // Destroy existing charts before creating new ones.
    [overallChartRef, distributionChartRef].forEach(ref => {
      if (ref.current) {
        const existingChart = Chart.getChart(ref.current);
        if (existingChart) existingChart.destroy();
      }
    });
  
    // --- First Chart: Overall Concept Distribution ---
    const overallCtx = overallChartRef.current?.getContext('2d');
    if (overallCtx) {
      if (clustersData.length > 0) {
        const clusterLabels = clustersData.map(cluster => cluster.label);
        const clusterFrequencies = clustersData.map(cluster => cluster.total_frequency);
        new Chart(overallCtx, {
          type: 'bar',
          data: {
            labels: clusterLabels,
            datasets: [{
              label: 'Cluster Frequency',
              data: clusterFrequencies,
              backgroundColor: 'rgba(75, 192, 192, 0.6)'
            }]
          },
          options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Overall Concept Distribution (By Cluster)' } },
            scales: {
              x: { beginAtZero: true },
              y: { beginAtZero: true }
            }
          }
        });
      } else {
        new Chart(overallCtx, {
          type: 'bar',
          data: {
            labels: Array.from(conceptData.concepts.keys()),
            datasets: [{
              label: 'Total Frequency',
              data: Array.from(conceptData.concepts.values()),
              backgroundColor: 'rgba(75, 192, 192, 0.6)'
            }]
          },
          options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Overall Concept Distribution' } },
            scales: {
              x: { beginAtZero: true },
              y: { beginAtZero: true }
            }
          }
        });
      }
    }
  
    // --- Second Chart: Clusters by Demographic ---
    const distributionCtx = distributionChartRef.current?.getContext('2d');
    if (distributionCtx && selectedGroup && conceptData.clusters && conceptData.clusters.demographics) {
      const clusterLabels = clustersData.map(cluster => cluster.label);
      
      // Filter keys: either "baseline" or those starting with the selectedGroup prefix.
      const relevantKeys = Object.keys(conceptData.clusters.demographics).filter(key =>
        key === "baseline" || key.includes(`${selectedGroup}:`)
      );
      
      const datasets = relevantKeys.map(key => {
        const demoClusters = conceptData.clusters.demographics[key];
        // For each overall cluster, find the corresponding demographic frequency.
        const data = clustersData.map(cluster => {
          const match = demoClusters.find(dc => dc.id === cluster.id);
          return match ? match.total_frequency : 0;
        });
        // Use a simplified label for subgroup keys.
        let displayLabel = key;
        if (key !== "baseline" && key.includes(':')) {
          displayLabel = key.split(':')[1];
        }
        // Use a fixed color for baseline, otherwise generate a color.
        const backgroundColor = getColorForAttribute(displayLabel);
        return {
          label: displayLabel,
          data,
          backgroundColor
        };
      });
  
      new Chart(distributionCtx, {
        type: 'bar',
        data: {
          labels: clusterLabels,
          datasets
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: `Clusters by ${selectedGroup}` }
          },
          scales: {
            y: { beginAtZero: true, title: { display: true, text: 'Total Frequency' } }
          }
        }
      });
    }
  }, [conceptData, selectedGroup, clustersData]);
  

  const handleDownloadCSV = () => {
    if (!conceptData.rawResults || !conceptData.extractedConcepts) {
      console.error("Missing required data for CSV export");
      return;
    }
    const csv = createConceptExtractionCSV(
      conceptData.rawResults,
      conceptData.extractedConcepts, 
      clustersData
    );
    downloadCSV(csv, 'concept_extraction_results.csv');
  };

  return (
    <div className="grid grid-cols-1 gap-6">
      <div className="flex justify-end">
        <Button
          onClick={handleDownloadCSV}
          disabled={!conceptData.rawResults || !conceptData.extractedConcepts}
          className="flex items-center gap-2"
        >
          <DownloadIcon className="h-4 w-4" />
          Categorized Data
        </Button>
      </div>

      <Card>
        <CardContent className="pt-4">
          <h3 className="text-lg font-semibold mb-4">Overall Concept Distribution</h3>
          <canvas ref={overallChartRef} />
        </CardContent>
      </Card>

      {demographicCategories.length > 0 && (
        <div className="flex items-center gap-2">
          <label htmlFor="demographic-select" className="font-medium">
            Filter by Demographic Type:
          </label>
          <select
            id="demographic-select"
            value={selectedGroup}
            onChange={(e) => setSelectedGroup(e.target.value)}
            className="border rounded p-1"
          >
            {demographicCategories.map(group => (
              <option key={group} value={group}>{group}</option>
            ))}
          </select>
        </div>
      )}

      <Card>
        <CardContent className="pt-4">
          <h3 className="text-lg font-semibold mb-4">
            Clusters by {selectedGroup}
          </h3>
          <canvas ref={distributionChartRef} />
        </CardContent>
      </Card>
    </div>
  );
}
