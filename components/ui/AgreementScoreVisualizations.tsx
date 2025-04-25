'use client';

import { useState } from 'react';
import { AgreementScores, AgreementVisualizationPoint, ContingencyTable } from '@/app/types/pipeline';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { Card, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type AgreementScoreVisualizationsProps = {
  agreementData: AgreementScores | null;
  embeddingsData?: { pca_one: number; pca_two: number }[];
};

type HeatmapViewProps = {
  tableData: ContingencyTable;
  title: string;
  xLabel: string; 
  yLabel: string; 
};


const CompactHeatmapView = ({ tableData, title, xLabel, yLabel }: HeatmapViewProps) => {
  const maxVal = Math.max(...tableData.table.flat());

  const getCellColor = (value: number) => {
    const intensity = maxVal ? Math.round((value / maxVal) * 255) : 0;
    return `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
  };

  return (
    <Card className="w-full">
      <CardContent className="p-4">
        <h3 className="text-base font-medium mb-3">{title}</h3>

        <div className="overflow-x-auto">
          <Table className="table-auto w-full border-collapse">
            <TableHeader>
              <TableRow>
                <TableHead className="border border-gray-300 bg-gray-100" />
                <TableHead
                  colSpan={tableData.colLabels.length}
                  className="border border-gray-300 bg-gray-100 text-center font-medium py-2"
                >
                  {xLabel} 
                </TableHead>
              </TableRow>

              <TableRow>
                <TableHead
                  rowSpan={tableData.rowLabels.length + 1}
                  className="border border-gray-300 bg-gray-100 text-center font-medium px-2 py-1"
                >
                  {yLabel} 
                </TableHead>
                {tableData.colLabels.map((colLabel, colIndex) => (
                  <TableHead
                    key={colIndex}
                    className="border border-gray-300 bg-gray-100 text-center px-2 py-1"
                    style={{ width: '4rem' }}
                  >
                    {colLabel}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>

            <TableBody>
              {tableData.rowLabels.map((rowLabel, rowIndex) => (
                <TableRow key={rowIndex}>
                  <TableCell className="border border-gray-300 bg-gray-100 text-center font-medium px-2 py-1"
                  style={{width: '4rem'}}
                  >
                    {rowLabel}
                  </TableCell>

                  {tableData.table[rowIndex].map((cell, cellIndex) => (
                    <TableCell
                      key={cellIndex}
                      className="border border-gray-300 text-center px-2 py-1"
                      style={{ backgroundColor: getCellColor(cell), width: '4rem' }}
                    >
                      {cell}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
};

type CustomShapeProps = {
  cx: number;
  cy: number;
  payload: AgreementVisualizationPoint & { aggregateAgreement: number };
};

const CombinedScatterPlot = ({ data }: { data: AgreementVisualizationPoint[] }) => {
  const enhancedData = data.map((d) => {
    const aggregateAgreement = d.cluster_topic_agree + d.cluster_pca_agree + d.topic_pca_agree;
    return { ...d, aggregateAgreement };
  });

  const getColor = (score: number) => {
    if (score === 3) return "#006400"; 
    if (score === 2) return "#32CD32"; 
    if (score === 1) return "#FFA500"; 
    return "#DC143C";
  };

  return (
    <div className="h-[400px] w-full p-4 border rounded-lg">
      <h3 className="text-sm font-medium mb-2">Combined Agreement Visualization</h3>
      <ResponsiveContainer width="100%" height={350}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 40, left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="pca_one"
            type="number"
            name="PCA Dimension 1"
            label={{ value: "PCA Dimension 1", position: "bottom", fill: "#333" }}
          />
          <YAxis
            dataKey="pca_two"
            type="number"
            name="PCA Dimension 2"
            label={{ value: "PCA Dimension 2", angle: -90, position: "insideLeft", fill: "#333" }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const point = payload[0].payload as AgreementVisualizationPoint & { aggregateAgreement: number };
                return (
                  <div className="bg-white p-2 border rounded">
                    <div><strong>PCA 1:</strong> {point.pca_one.toFixed(2)}</div>
                    <div><strong>PCA 2:</strong> {point.pca_two.toFixed(2)}</div>
                    <div><strong>Cluster-Topic:</strong> {point.cluster_topic_agree ? 'Agree' : 'Disagree'}</div>
                    <div><strong>Cluster-Embedding:</strong> {point.cluster_pca_agree ? 'Agree' : 'Disagree'}</div>
                    <div><strong>Topic-Embedding:</strong> {point.topic_pca_agree ? 'Agree' : 'Disagree'}</div>
                    <div><strong>Aggregate:</strong> {point.aggregateAgreement} / 3</div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Scatter
            name="Data Points"
            data={enhancedData}
            fill="#8884d8"
            shape={(props: unknown) => {
              const customProps = props as CustomShapeProps;
              const { cx, cy, payload } = customProps;
              const color = getColor(payload.aggregateAgreement);
              return (
                <circle cx={cx} cy={cy} r={5} fill={color} stroke="#333" strokeWidth={1} />
              );
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export function AgreementScoreVisualizations({ agreementData }: AgreementScoreVisualizationsProps) {
  const [heatmapType, setHeatmapType] = useState<'cluster_topic' | 'cluster_pca' | 'topic_pca'>('cluster_topic');

  if (!agreementData) {
    return (
      <div className="flex items-center justify-center h-48">
        <p className="text-muted-foreground">Calculating agreement scores...</p>
      </div>
    );
  }

  let heatmapData: ContingencyTable;
  let heatmapTitle: string;
  let xLabel: string;
  let yLabel: string;
  switch (heatmapType) {
    case 'cluster_topic':
      heatmapData = agreementData.mapping_data.contingency_tables.cluster_topic;
      heatmapTitle = 'Cluster-Topic Heatmap';
      xLabel = 'Dominant Topic';   
      yLabel = 'Concept Clusters'; 
      break;
    case 'cluster_pca':
      heatmapData = agreementData.mapping_data.contingency_tables.cluster_pca;
      heatmapTitle = 'Cluster-Embeddings Heatmap';
      xLabel = 'Embeddings';
      yLabel = 'Concept Clusters';
      break;
    case 'topic_pca':
      heatmapData = agreementData.mapping_data.contingency_tables.topic_pca;
      heatmapTitle = 'Topic-Embeddings Heatmap';
      xLabel = 'Embeddings Cluster';
      yLabel = 'Dominant Topic';
      break;
    default:
      heatmapData = agreementData.mapping_data.contingency_tables.cluster_topic;
      heatmapTitle = 'Cluster-Topic Heatmap';
      xLabel = 'Topics';
      yLabel = 'Clusters';
      break;
  }

  return (
    <div className="space-y-6">
      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-lg border bg-card">
        <div>
          <p className="font-medium">Cluster-Topic Agreement:</p>
          <p className="text-lg">{(agreementData.agreement_scores.cluster_topic * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="font-medium">Cluster-Embedding Agreement:</p>
          <p className="text-lg">{(agreementData.agreement_scores.cluster_embedding * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="font-medium">Topic-Embedding Agreement:</p>
          <p className="text-lg">{(agreementData.agreement_scores.topic_embedding * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Combined Scatter Plot */}
      <CombinedScatterPlot data={agreementData.visualization_data} />

      {/* Heatmap Toggle */}
      <div className="flex gap-2 justify-center">
        <button
          className={`px-3 py-1 rounded ${
            heatmapType === 'cluster_topic' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
          onClick={() => setHeatmapType('cluster_topic')}
        >
          Cluster-Topic
        </button>
        <button
          className={`px-3 py-1 rounded ${
            heatmapType === 'cluster_pca' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
          onClick={() => setHeatmapType('cluster_pca')}
        >
          Cluster-Embedding
        </button>
        <button
          className={`px-3 py-1 rounded ${
            heatmapType === 'topic_pca' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
          onClick={() => setHeatmapType('topic_pca')}
        >
          Topic-Embedding
        </button>
      </div>

      {/* Compact Heatmap */}
      <CompactHeatmapView
        tableData={heatmapData}
        title={heatmapTitle}
        xLabel={xLabel}
        yLabel={yLabel}
      />

      {/* Mapping Views */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="w-full">
          <CardContent className="p-4">
            <h3 className="text-sm font-medium mb-4">Cluster to Topic Mapping</h3>
            <div className="space-y-2">
              {Object.entries(agreementData.mapping_data.cluster_topic_mapping).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center text-sm">
                  <span>{key}</span>
                  <span>→</span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card className="w-full">
          <CardContent className="p-4">
            <h3 className="text-sm font-medium mb-4">Cluster to Embedding Mapping</h3>
            <div className="space-y-2">
              {Object.entries(agreementData.mapping_data.cluster_pca_mapping).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center text-sm">
                  <span>{key}</span>
                  <span>→</span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card className="w-full">
          <CardContent className="p-4">
            <h3 className="text-sm font-medium mb-4">Topic to Embedding Mapping</h3>
            <div className="space-y-2">
              {Object.entries(agreementData.mapping_data.topic_pca_mapping).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center text-sm">
                  <span>{key}</span>
                  <span>→</span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}