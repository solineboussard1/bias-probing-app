import { Card } from "./card";
import { Badge } from "./badge";
import { useState } from "react";
import { Button } from "./button";
import { ChevronDown, ChevronUp, Download } from "lucide-react";
import { ResponsiveScatterPlot } from '@nivo/scatterplot';
import { downloadCSV, createEmbeddingsExtractionCSV } from "@/app/lib/csv-utils";
import { AnalysisResult } from "@/app/types/pipeline";

type EmbeddingsResult = {
    cluster_id: number;
    size: number;
    representative_responses: string[];
    distribution: { [key: string]: number };
    coordinates: number[][];
    embeddings: number[][];
};

type EmbeddingsVisualizationsProps = {
    results: EmbeddingsResult[];
    analysisResults?: AnalysisResult[];
}


const ITEMS_PER_PAGE = 5;

export function EmbeddingsVisualizations({ results, analysisResults }: EmbeddingsVisualizationsProps) {
    const [pagination, setPagination] = useState<{
        [key: number]: {
            page: number;
            expanded: Set<number>;
        };
    }>({});

    // 1) Build response → demographic lookup
    const responseToDemo = new Map<string, string>();
    analysisResults?.forEach(ar =>
        ar.prompts.forEach(p =>
            p.responses.forEach(resp => {
                const demo = p.metadata.demographics?.[0] ?? 'baseline';
                responseToDemo.set(resp, demo.replace(/^default:/, ''));
            })
        )
    );

    // 2) Group each PCA point by its demographic
    const grouped: Record<string, { id: string; data: any[] }> = {};
    results.forEach(cluster =>
        cluster.coordinates.forEach((coord, idx) => {
            const resp = cluster.representative_responses[idx];
            const demo = responseToDemo.get(resp) || 'baseline';
            if (!grouped[demo]) grouped[demo] = { id: demo, data: [] };
            grouped[demo].data.push({
                x: coord[0],
                y: coord[1],
                clusterId: cluster.cluster_id + 1,
                response: resp
            });
        })
    );

    // 3) Create the array Nivo expects
    const scatterData = Object.values(grouped);


    return (
        <div className="space-y-6">
            {/* Download Button */}
            <div className="flex justify-end">
                <Button
                    onClick={() => {
                        if (!analysisResults) {
                            console.error("Missing required data for embeddings CSV export");
                            return;
                        }
                        const csv = createEmbeddingsExtractionCSV(analysisResults, results);
                        downloadCSV(csv, 'data_with_pca.csv');
                    }}
                    className="flex items-center gap-2"
                    disabled={!analysisResults}
                >
                    <Download className="h-4 w-4" />
                    Embedding Data
                </Button>
            </div>

            {/* PCA Visualization */}
            <Card className="p-4">
                <h4 className="font-medium mb-4">PCA by Demographic Group</h4>
                <div style={{ height: '400px' }}>
                    <ResponsiveScatterPlot
                        data={scatterData}
                        useMesh={false}
                        margin={{ top: 20, right: 20, bottom: 70, left: 70 }}
                        xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                        yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                        axisTop={null}
                        axisRight={null}
                        axisBottom={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: 'PC1',
                            legendPosition: 'middle',
                            legendOffset: 46
                        }}
                        axisLeft={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: 'PC2',
                            legendPosition: 'middle',
                            legendOffset: -60
                        }}
                        legends={[
                            {
                                anchor: 'bottom-right',
                                direction: 'column',
                                translateX: -10,
                                translateY: -10,
                                itemWidth: 80,
                                itemHeight: 20,
                                symbolSize: 12,
                                symbolShape: 'circle',
                            }
                        ]}
                        colors={{ scheme: 'category10' }}
                        tooltip={({ node }) => {
                            if (
                                node?.data?.x === undefined ||
                                node?.data?.y === undefined ||
                                node?.data?.response === undefined
                            ) {
                                return null;
                            }
                            return (
                                <div className="bg-white p-2 shadow-lg rounded-lg border text-sm">
                                    <strong>Demographic:</strong> {node.serieId}
                                    <br />
                                    <strong>Cluster:</strong> {node.data.clusterId}
                                </div>
                            );
                        }}                        
                    />
                </div>
            </Card>

            {/* Cluster Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {results.map((cluster) => (
                    <Card key={cluster.cluster_id} className="p-4">
                        <div className="space-y-4">
                            <div className="flex justify-between items-center">
                                <h4 className="font-medium">Cluster {cluster.cluster_id + 1}</h4>
                                <Badge variant="secondary">Size: {cluster.size}</Badge>
                            </div>

                            <div>
                                <h5 className="text-sm font-medium mb-2">Distribution:</h5>
                                <div className="flex flex-wrap gap-2">
                                    {Object.entries(cluster.distribution).map(([key, value]) => {
                                        // Remove 'default:' prefix, brackets and quotes from the key.
                                        const cleanedKey = key.replace(/default:|\[|\]|'/g, '');
                                        return (
                                            <Badge key={key} variant="outline">
                                                {cleanedKey}: {value}
                                            </Badge>
                                        );
                                    })}
                                </div>
                            </div>

                            <div className="space-y-2">
                                <h5 className="font-medium text-sm tracking-tight pb-1 border-b">Responses:</h5>
                                {cluster.representative_responses
                                    .slice(
                                        (pagination[cluster.cluster_id]?.page || 0) * ITEMS_PER_PAGE,
                                        ((pagination[cluster.cluster_id]?.page || 0) + 1) * ITEMS_PER_PAGE
                                    )
                                    .map((response, idx) => {
                                        const absoluteIdx = idx + (pagination[cluster.cluster_id]?.page || 0) * ITEMS_PER_PAGE;
                                        const isExpanded = pagination[cluster.cluster_id]?.expanded?.has(absoluteIdx);

                                        return (
                                            <div key={idx} className="border rounded-lg p-3">
                                                <div
                                                    className="flex items-center justify-between cursor-pointer"
                                                    onClick={() => {
                                                        setPagination(prev => {
                                                            const currentExpanded = new Set(prev[cluster.cluster_id]?.expanded || []);
                                                            if (isExpanded) {
                                                                currentExpanded.delete(absoluteIdx);
                                                            } else {
                                                                currentExpanded.add(absoluteIdx);
                                                            }
                                                            return {
                                                                ...prev,
                                                                [cluster.cluster_id]: {
                                                                    page: prev[cluster.cluster_id]?.page || 0,
                                                                    expanded: currentExpanded
                                                                }
                                                            };
                                                        });
                                                    }}
                                                >
                                                    <p className="text-sm font-medium">{response.slice(0, 50) + (response.length > 50 ? '...' : '')}</p>
                                                    {isExpanded ? (
                                                        <ChevronUp className="h-4 w-4 flex-shrink-0" />
                                                    ) : (
                                                        <ChevronDown className="h-4 w-4 flex-shrink-0" />
                                                    )}
                                                </div>

                                                {isExpanded && (
                                                    <div className="mt-2">
                                                        <p className="text-sm text-muted-foreground">{response}</p>
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}

                                {/* Pagination Controls */}
                                {cluster.representative_responses.length > ITEMS_PER_PAGE && (
                                    <div className="flex justify-center gap-2 mt-3 pt-2 border-t">
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => {
                                                setPagination(prev => ({
                                                    ...prev,
                                                    [cluster.cluster_id]: {
                                                        page: Math.max(0, (prev[cluster.cluster_id]?.page || 0) - 1),
                                                        expanded: prev[cluster.cluster_id]?.expanded || new Set()
                                                    }
                                                }));
                                            }}
                                            disabled={(pagination[cluster.cluster_id]?.page || 0) === 0}
                                        >
                                            Previous
                                        </Button>

                                        <span className="flex items-center text-sm text-muted-foreground">
                                            Page {(pagination[cluster.cluster_id]?.page || 0) + 1} of{' '}
                                            {Math.ceil(cluster.representative_responses.length / ITEMS_PER_PAGE)}
                                        </span>

                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => {
                                                setPagination(prev => ({
                                                    ...prev,
                                                    [cluster.cluster_id]: {
                                                        page: Math.min(
                                                            Math.ceil(cluster.representative_responses.length / ITEMS_PER_PAGE) - 1,
                                                            (prev[cluster.cluster_id]?.page || 0) + 1
                                                        ),
                                                        expanded: prev[cluster.cluster_id]?.expanded || new Set()
                                                    }
                                                }));
                                            }}
                                            disabled={
                                                (pagination[cluster.cluster_id]?.page || 0) >=
                                                Math.ceil(cluster.representative_responses.length / ITEMS_PER_PAGE) - 1
                                            }
                                        >
                                            Next
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </Card>
                ))}
            </div>
        </div>
    );
}
