import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function QuoteEditor() {
  const location = useLocation();
  const navigate = useNavigate();
  // Quote data passed via state or query param
  const quote = location.state?.quote || {};
  // Support both old (array of strings) and new (array of objects) item formats
  const [fields, setFields] = useState({
    vendor: quote.vendor || '',
    date: quote.date || '',
    total: quote.total || '',
    items: Array.isArray(quote.items) && quote.items.length > 0 && typeof quote.items[0] === 'object'
      ? quote.items
      : (Array.isArray(quote.items) ? quote.items.map((item: any) => ({ description: item, quantity: '', price: '', total: '' })) : []),
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFields({ ...fields, [e.target.name]: e.target.value });
  };

  const handleItemFieldChange = (idx: number, field: string, value: string) => {
    const items = [...fields.items];
    items[idx] = { ...items[idx], [field]: value };
    setFields({ ...fields, items });
  };

  const handleSave = () => {
    // TODO: Save quote to backend
    alert('Quote saved!');
    navigate(-1);
  };

  return (
    <div style={{ maxWidth: 700, margin: '40px auto', background: '#fff', borderRadius: 12, boxShadow: '0 2px 16px #eee', padding: 32 }}>
      <h2>Edit Quote</h2>
      <label>Vendor/Customer:<br />
        <input name="vendor" value={fields.vendor} onChange={handleChange} style={{ width: '100%' }} />
      </label>
      <br /><br />
      <label>Date:<br />
        <input name="date" value={fields.date} onChange={handleChange} style={{ width: '100%' }} />
      </label>
      <br /><br />
      <label>Total:<br />
        <input name="total" value={fields.total} onChange={handleChange} style={{ width: '100%' }} />
      </label>
      <br /><br />
      <label>Items:</label>
      {fields.items.length === 0 ? (
        <div style={{ color: '#888', margin: '12px 0' }}>No items detected in the quote. You can add them manually.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
            <thead>
              <tr style={{ background: '#f5f5f5' }}>
                <th style={{ padding: 8, border: '1px solid #eee' }}>Description</th>
                <th style={{ padding: 8, border: '1px solid #eee' }}>Quantity</th>
                <th style={{ padding: 8, border: '1px solid #eee' }}>Price</th>
                <th style={{ padding: 8, border: '1px solid #eee' }}>Total</th>
              </tr>
            </thead>
            <tbody>
              {fields.items.map((item: any, idx: number) => (
                <tr key={idx}>
                  <td style={{ padding: 8, border: '1px solid #eee' }}>
                    <input
                      value={item.description || ''}
                      onChange={e => handleItemFieldChange(idx, 'description', e.target.value)}
                      style={{ width: '100%' }}
                    />
                  </td>
                  <td style={{ padding: 8, border: '1px solid #eee' }}>
                    <input
                      value={item.quantity || ''}
                      onChange={e => handleItemFieldChange(idx, 'quantity', e.target.value)}
                      style={{ width: '100%' }}
                    />
                  </td>
                  <td style={{ padding: 8, border: '1px solid #eee' }}>
                    <input
                      value={item.price || ''}
                      onChange={e => handleItemFieldChange(idx, 'price', e.target.value)}
                      style={{ width: '100%' }}
                    />
                  </td>
                  <td style={{ padding: 8, border: '1px solid #eee' }}>
                    <input
                      value={item.total || ''}
                      onChange={e => handleItemFieldChange(idx, 'total', e.target.value)}
                      style={{ width: '100%' }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <button onClick={handleSave} style={{ marginTop: 16, padding: '8px 24px', background: '#667eea', color: '#fff', border: 'none', borderRadius: 6 }}>Save</button>
    </div>
  );
}
