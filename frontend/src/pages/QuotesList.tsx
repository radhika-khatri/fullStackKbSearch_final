import React from 'react';

// Dummy data for now, replace with API call if needed
const quotes = [
  {
    id: 1,
    vendor: 'Micronix Infosolutions LTD',
    date: '13-01-2025',
    total: '116000.00',
    items: [
      { description: 'Dell Inspiron 3520', quantity: '1', price: '...', total: '...' },
      { description: 'Dell Latitude 3440', quantity: '1', price: '...', total: '...' },
    ]
  }
];

export default function QuotesList() {
  return (
    <div style={{ maxWidth: 900, margin: '40px auto', background: '#fff', borderRadius: 12, boxShadow: '0 2px 16px #eee', padding: 32 }}>
      <h2 style={{ color: '#667eea', marginBottom: 24 }}>Quotes</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', background: '#fafaff' }}>
        <thead style={{ background: '#f5f5f5' }}>
          <tr>
            <th style={{ padding: 10, border: '1px solid #eee' }}>Vendor</th>
            <th style={{ padding: 10, border: '1px solid #eee' }}>Date</th>
            <th style={{ padding: 10, border: '1px solid #eee' }}>Total</th>
            <th style={{ padding: 10, border: '1px solid #eee' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {quotes.map(q => (
            <tr key={q.id}>
              <td style={{ padding: 10, border: '1px solid #eee' }}>{q.vendor}</td>
              <td style={{ padding: 10, border: '1px solid #eee' }}>{q.date}</td>
              <td style={{ padding: 10, border: '1px solid #eee' }}>{q.total}</td>
              <td style={{ padding: 10, border: '1px solid #eee' }}>
                <a href={`/quote-editor?id=${q.id}`} style={{ color: '#667eea', textDecoration: 'underline' }}>View/Edit</a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
