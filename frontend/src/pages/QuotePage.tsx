
import React, { useState, useRef } from 'react';
import { FaEdit, FaSearchPlus, FaSearchMinus, FaSyncAlt } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import { chatWithImage } from '../services/api';
import axios from 'axios';
import '../pages/Chat.css';

const QuotePage: React.FC = () => {
  const [showForm, setShowForm] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [ocrText, setOcrText] = useState('');
  type QuoteTableRow = {
    description?: string;
    qty?: string;
    unit?: string;
    unit_price?: string;
    discount?: string;
    net_price?: string;
    [key: string]: any;
  };
  const [ocrTable, setOcrTable] = useState<QuoteTableRow[]>([]);
  const [tableError, setTableError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [quote, setQuote] = useState('');
  const [zoom, setZoom] = useState(1);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleCreateQuote = () => {
    setShowForm(true);
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        alert('Image file too large. Maximum size is 10MB.');
        return;
      }
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Only extract relevant fields for a quote (customize as needed)
  const RELEVANT_FIELDS = [
    'Category', 'Material', 'Quote', 'Description', 'Price', 'Amount', 'Quantity', 'Unit', 'Code', 'SKU', 'Product', 'Item', 'Total', 'Tax', 'Discount', 'Customer', 'Supplier', 'Date', 'Order', 'Invoice', 'Number', 'Name', 'Brand', 'Model', 'Color', 'Size', 'Weight', 'Contact', 'Phone', 'Email', 'Address', 'Notes', 'Remarks'
  ];

  const parseOcrToTable = (ocr: string): any[] => {
    if (!ocr) return [];
    // Try to extract only lines with relevant fields
    return ocr.split(/\n|\r/)
      .map(line => line.trim())
      .filter(Boolean)
      .map(line => {
        // Try to match "Field: Value" or "Field - Value"
        const match = line.match(/^([\w\s]+)[\s:=-]+(.+)$/);
        if (match) {
          const key = match[1].trim();
          const value = match[2].trim();
          if (RELEVANT_FIELDS.some(f => key.toLowerCase().includes(f.toLowerCase()))) {
            return { key, value };
          }
        }
        return null;
      })
      .filter(Boolean);
  };

  interface ParseTableResponse {
    table: QuoteTableRow[];
  }

  // Use Gemini for everything: send image, get table from Gemini response
  const handleUploadAndOcr = async () => {
    if (!selectedImage) return;
    setIsLoading(true);
    setOcrText('');
    setOcrTable([]);
    setQuote('');
    setTableError(null);
    try {
      // Send image to Gemini endpoint (chatWithImage)
      const res = await chatWithImage('', selectedImage);
      // Use only the 'table' field from backend response
      const table = (res.data && res.data.table) || [];
      setOcrText(''); // Don't show natural language
      if (Array.isArray(table) && table.length > 0) {
        setOcrTable(table);
        setTableError(null);
      } else {
        // Always show at least one empty row for user input
        setOcrTable([{ description: '', qty: '', unit: '', unit_price: '', discount: '', net_price: '' }]);
        setTableError(null);
      }
    } catch (err: any) {
      setOcrText('Failed to extract text from image.');
      setOcrTable([{ description: '', qty: '', unit: '', unit_price: '', discount: '', net_price: '' }]);
      setTableError('Failed to parse table from Gemini.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle table cell edit for any field
  const handleTableEdit = (idx: number, field: keyof QuoteTableRow, value: string) => {
    setOcrTable(prev => prev.map((row, i) => i === idx ? { ...row, [field]: value, editing: true } : row));
  };

  // Go to edit page for further editing
  const handleEditPage = () => {
    // Pass ocrTable and imagePreview as state (or use a global store)
    navigate('/quotes/edit', { state: { ocrTable, imagePreview } });
  };

  // Confirm quote from table
  const handleConfirmQuote = () => {
    // Join all key-value pairs as a string for the quote
    const quoteStr = ocrTable
      .filter(row => row.key || row.value)
      .map(row => row.key ? `${row.key}: ${row.value}` : row.value)
      .join(' | ');
    setQuote(quoteStr);
    setShowWarning(true);
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowWarning(true);
  };

  const handleWarningConfirm = () => {
    alert(`Quote template saved: ${quote}`);
    setShowWarning(false);
    setShowForm(false);
    setSelectedImage(null);
    setImagePreview(null);
    setOcrText('');
    setOcrTable([]);
    setQuote('');
  };

  const handleWarningCancel = () => {
    setShowWarning(false);
  };

  return (
    <div>
      <h1 style={{ color: 'red', textAlign: 'center', margin: 16 }}>Quote Page Loaded</h1>
      <div className="chat-layout" style={{ justifyContent: 'center', alignItems: 'flex-start', minHeight: '100vh' }}>
        <div className="chat-main" style={{ maxWidth: 1200, margin: '40px auto', width: '100%' }}>
          <h2 style={{ textAlign: 'center', marginBottom: 24 }}>Quote Generator</h2>
          {/* ...existing code... */}
          {!showForm && (
            <button className="new-chat" style={{ width: '100%' }} onClick={handleCreateQuote}>
              Create New Quote
            </button>
          )}
          {showForm && (
            <form className="chat-input" style={{ flexDirection: 'row', gap: 32, alignItems: 'flex-start', background: 'none', border: 'none', boxShadow: 'none' }} onSubmit={handleFormSubmit}>
              {/* Left: Image upload/preview with controls */}
              <div style={{ flex: 1, minWidth: 320, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  style={{ display: 'none' }}
                />
                <button
                  type="button"
                  className="image-upload-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  style={{ width: '100%', marginBottom: 16 }}
                >
                  {selectedImage ? 'Change Image' : 'Upload Image'}
                </button>
                {imagePreview && (
                  <div className="image-preview-section" style={{ marginBottom: 16, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <div className="image-preview-container" style={{ justifyContent: 'center', position: 'relative' }}>
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="image-preview"
                        style={{ maxWidth: 400 * zoom, maxHeight: 400 * zoom, border: '1px solid #ccc', borderRadius: 8, transition: 'all 0.2s' }}
                      />
                      <div style={{ display: 'flex', gap: 8, marginTop: 8, justifyContent: 'center' }}>
                        <button type="button" className="image-upload-btn" title="Zoom In" onClick={() => setZoom(z => Math.min(z + 0.2, 2))}><FaSearchPlus /></button>
                        <button type="button" className="image-upload-btn" title="Zoom Out" onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))}><FaSearchMinus /></button>
                        <button type="button" className="image-upload-btn" title="Reset Zoom" onClick={() => setZoom(1)}><FaSyncAlt /></button>
                      </div>
                    </div>
                    <span className="image-preview-text">Image selected</span>
                  </div>
                )}
                {selectedImage && !ocrText && (
                  <button
                    type="button"
                    className="send-btn"
                    onClick={handleUploadAndOcr}
                    disabled={isLoading}
                    style={{ width: '100%' }}
                  >
                    {isLoading ? 'Reading...' : 'Extract Text from Image'}
                  </button>
                )}
              </div>
              {/* Right: Structured Table Output */}
              <div style={{ flex: 2, minWidth: 400 }}>
                {tableError && (
                  <div style={{ color: 'red', marginBottom: 12 }}>{tableError}</div>
                )}
                {ocrTable.length > 0 && (
                  <>
                    <div style={{ marginBottom: 16, color: '#666', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <b>Extracted Table (edit as needed):</b>
                      <button type="button" className="image-upload-btn" style={{ fontSize: 18 }} onClick={handleEditPage} title="Edit in full page"><FaEdit /> Edit Page</button>
                    </div>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr style={{ background: '#f5f5f5' }}>
                          <th style={{ padding: 8, border: '1px solid #eee' }}>Description</th>
                          <th style={{ padding: 8, border: '1px solid #eee' }}>Qty</th>
                          <th style={{ padding: 8, border: '1px solid #eee' }}>Unit</th>
                          <th style={{ padding: 8, border: '1px solid #eee' }}>Unit Price</th>
                          <th style={{ padding: 8, border: '1px solid #eee' }}>Discount</th>
                          <th style={{ padding: 8, border: '1px solid #eee' }}>Net Price</th>
                          <th style={{ padding: 8, border: '1px solid #eee' }}>Edit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ocrTable.map((row, idx) => (
                          <tr key={idx}>
                            <td style={{ padding: 8, border: '1px solid #eee' }}>
                              <input
                                type="text"
                                value={row.description ?? ''}
                                onChange={e => handleTableEdit(idx, 'description', e.target.value)}
                                style={{ width: '100%', border: 'none', background: 'transparent' }}
                                placeholder="Description"
                              />
                            </td>
                            <td style={{ padding: 8, border: '1px solid #eee' }}>
                              <input
                                type="text"
                                value={row.qty ?? ''}
                                onChange={e => handleTableEdit(idx, 'qty', e.target.value)}
                                style={{ width: '100%', border: 'none', background: 'transparent' }}
                                placeholder="Qty"
                              />
                            </td>
                            <td style={{ padding: 8, border: '1px solid #eee' }}>
                              <input
                                type="text"
                                value={row.unit ?? ''}
                                onChange={e => handleTableEdit(idx, 'unit', e.target.value)}
                                style={{ width: '100%', border: 'none', background: 'transparent' }}
                                placeholder="Unit"
                              />
                            </td>
                            <td style={{ padding: 8, border: '1px solid #eee' }}>
                              <input
                                type="text"
                                value={row.unit_price ?? ''}
                                onChange={e => handleTableEdit(idx, 'unit_price', e.target.value)}
                                style={{ width: '100%', border: 'none', background: 'transparent' }}
                                placeholder="Unit Price"
                              />
                            </td>
                            <td style={{ padding: 8, border: '1px solid #eee' }}>
                              <input
                                type="text"
                                value={row.discount ?? ''}
                                onChange={e => handleTableEdit(idx, 'discount', e.target.value)}
                                style={{ width: '100%', border: 'none', background: 'transparent' }}
                                placeholder="Discount"
                              />
                            </td>
                            <td style={{ padding: 8, border: '1px solid #eee' }}>
                              <input
                                type="text"
                                value={row.net_price ?? ''}
                                onChange={e => handleTableEdit(idx, 'net_price', e.target.value)}
                                style={{ width: '100%', border: 'none', background: 'transparent' }}
                                placeholder="Net Price"
                              />
                            </td>
                            <td style={{ padding: 8, border: '1px solid #eee', textAlign: 'center' }}>
                              <FaEdit style={{ cursor: 'pointer', color: '#6f42c1' }} title="Edit" onClick={() => {}} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <button
                      type="button"
                      className="send-btn"
                      style={{ marginBottom: 12 }}
                      onClick={handleConfirmQuote}
                    >
                      Confirm & Set Quote
                    </button>
                  </>
                )}
                {quote && (
                  <div style={{ margin: '12px 0', color: '#333', textAlign: 'center', fontWeight: 500 }}>
                    <div>Final Quote:</div>
                    <div style={{ fontStyle: 'italic', marginTop: 4 }}>{quote}</div>
                  </div>
                )}
                {quote && (
                  <button type="submit" className="send-btn" style={{ width: '100%' }}>
                    Save Quote Template
                  </button>
                )}
              </div>
            </form>
          )}
          {/* Warning Popup */}
          {showWarning && (
            <div style={{
              position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
              background: 'rgba(0,0,0,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
            }}>
              <div style={{ background: '#fff', borderRadius: 12, padding: 32, minWidth: 320, boxShadow: '0 4px 24px rgba(0,0,0,0.15)' }}>
                <div style={{ fontWeight: 600, fontSize: 18, marginBottom: 16, color: '#d9534f' }}>Warning</div>
                <div style={{ marginBottom: 20 }}>Are you sure you want to save this quote template? Please confirm all details are correct.</div>
                <div style={{ display: 'flex', gap: 16, justifyContent: 'center' }}>
                  <button className="send-btn" style={{ background: '#28a745' }} onClick={handleWarningConfirm} type="button">Yes, Save</button>
                  <button className="send-btn" style={{ background: '#dc3545' }} onClick={handleWarningCancel} type="button">Cancel</button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default QuotePage;
