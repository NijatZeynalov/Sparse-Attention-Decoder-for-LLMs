
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class MetricsVisualizer:
    def __init__(self, save_dir=None, color_palette=None):
        self.save_dir = save_dir
        self.color_palette = color_palette or ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    def create_metrics_overview(self, metrics, save=True):
        try:
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Throughput', 'Memory Usage', 'Sparsity', 'Cache Performance'))
            
            # Throughput
            throughput = metrics['performance'].get('throughput_history', [])
            fig.add_trace(
                go.Scatter(
                    y=throughput,
                    name='Throughput',
                    line=dict(color=self.color_palette[0])
                ),
                row=1,
                col=1
            )

            # Memory usage
            memory = metrics['performance'].get('memory_history', [])
            fig.add_trace(
                go.Scatter(
                    y=memory,
                    name='Memory (MB)',
                    line=dict(color=self.color_palette[1])
                ),
                row=1,
                col=2
            )

            # Sparsity across layers
            sparsity = [
                metrics['attention'][f'layer_{i}']['sparsity']
                for i in range(len(metrics['attention']))
            ]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sparsity))),
                    y=sparsity,
                    name='Sparsity',
                    marker_color=self.color_palette[2]
                ),
                row=2,
                col=1
            )

            # Cache performance
            cache_stats = metrics['cache']
            fig.add_trace(
                go.Pie(
                    labels=['Hits', 'Misses'],
                    values=[cache_stats['hits'], cache_stats['misses']],
                    marker_colors=[self.color_palette[3], self.color_palette[4]]
                ),
                row=2,
                col=2
            )

            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Sparse Attention Metrics Overview"
            )

            if save and self.save_dir:
                fig.write_html(self.save_dir / 'metrics_overview.html')

        except Exception as e:
            logger.error(f"Error creating metrics overview: {str(e)}")
