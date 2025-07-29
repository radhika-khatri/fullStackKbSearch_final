import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

export function exportToCSV(data: any[], filename: string) {
  const csv = data.map(row => Object.values(row).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.setAttribute('download', `${filename}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function exportToPDF(containerId: string, filename: string) {
  const doc = new jsPDF();
  const element = document.getElementById(containerId);

  if (element) {
    autoTable(doc, { html: `#${containerId} table` });
    doc.save(`${filename}.pdf`);
  } else {
    console.error(`Element with ID '${containerId}' not found`);
  }
}
