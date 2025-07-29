import React from "react";
import { BiTrendingUp, BiTrendingDown, BiInfoCircle, BiDownload } from "react-icons/bi";
import { exportToPDF } from "../utils/exportUtils";

const insights = [
  {
    title: "Highest CTR",
    value: "Email – 46.84%",
    type: "positive",
    tooltip: "Click-through rate is the % of recipients who clicked a link.",
  },
  {
    title: "Top Revenue Channel",
    value: "Email – $61.28",
    type: "positive",
    tooltip: "Revenue earned from the most profitable channel.",
  },
  {
    title: "Best Day",
    value: "July 5, 2025",
    type: "neutral",
    tooltip: "The day with the highest overall performance.",
  },
];

export default function InsightCards() {
  const handleExport = () => exportToPDF("insights-table-wrapper", "insights");

  const renderIcon = (type: string) => {
    if (type === "positive") return <BiTrendingUp className="icon positive" />;
    if (type === "negative") return <BiTrendingDown className="icon negative" />;
    return <BiInfoCircle className="icon neutral" />;
  };

  return (
    <section className="insight-section glass">
      <div className="insight-header">
        <h2 className="insight-heading">Business Performance Highlights</h2>
        <button onClick={handleExport} className="export-button" title="Download PDF">
          <BiDownload /> Export
        </button>
      </div>

      <div className="insight-grid">
        {insights.map((insight, idx) => (
          <div className="insight-card glass" key={idx} title={insight.tooltip}>
            {renderIcon(insight.type)}
            <div className="insight-content">
              <h4 className="insight-title">{insight.title}</h4>
              <p className="insight-value">{insight.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Hidden but exportable DOM target */}
      <div id="insights-table-wrapper" style={{ display: "none" }}>
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {insights.map((i, idx) => (
              <tr key={idx}>
                <td>{i.title}</td>
                <td>{i.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
