import React, { useState } from 'react';
import './TemplatePage.css';
import { FaEdit, FaTrash } from 'react-icons/fa';

const predefinedTemplates = [
  {
    type: 'whatsapp',
    title: 'Welcome Message',
    description: 'Greet new users with a warm hug 🥰'
  },
  {
    type: 'email',
    title: 'Limited-Time Offer',
    description: 'Flash sale hype builder'
  },
  {
    type: 'whatsapp',
    title: 'Follow-Up Message',
    description: 'Check in with users after their first interaction'
  },
  {
    type: 'email',
    title: 'Newsletter',
    description: 'Monthly updates and news'
  }
];

export default function TemplatePage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [customTemplates, setCustomTemplates] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newTemplate, setNewTemplate] = useState({ title: '', description: '', type: 'whatsapp' });
  const [editTemplateIndex, setEditTemplateIndex] = useState(null);

  const filteredTemplates = [...predefinedTemplates, ...customTemplates].filter((template) => {
    const matchesSearch = template.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || template.type === filterType;
    return matchesSearch && matchesFilter;
  });

  const handleTemplateClick = (template) => {
    alert(`You selected the template: ${template.title}`);
  };

  const handleCreateOrEditTemplate = () => {
    if (newTemplate.title && newTemplate.description && (newTemplate.type === 'whatsapp' || newTemplate.type === 'email')) {
      if (editTemplateIndex !== null) {
        const updatedTemplates = [...customTemplates];
        updatedTemplates[editTemplateIndex] = newTemplate;
        setCustomTemplates(updatedTemplates);
        setEditTemplateIndex(null);
      } else {
        setCustomTemplates([...customTemplates, newTemplate]);
      }
      setNewTemplate({ title: '', description: '', type: 'whatsapp' });
      setIsModalOpen(false);
      alert('Template saved successfully!');
    } else {
      alert('Invalid input. Please fill all fields correctly.');
    }
  };

  const handleEditTemplate = (index) => {
    const templateToEdit = index < predefinedTemplates.length
      ? predefinedTemplates[index]
      : customTemplates[index - predefinedTemplates.length];

    if (templateToEdit) {
      setNewTemplate(templateToEdit);
      setEditTemplateIndex(index - predefinedTemplates.length);
      setIsModalOpen(true);
    } else {
      alert('Template not found.');
    }
  };

  const handleDeleteTemplate = (index) => {
    const updatedTemplates = customTemplates.filter((_, i) => i !== index);
    setCustomTemplates(updatedTemplates);
    alert('Template deleted successfully!');
  };

  return (
    <div className="template-page">
      <header className="template-header">
        <div>
          <h1>Template Builder</h1>
          <p>Create & preview your WhatsApp / Email templates effortlessly</p>
        </div>

        <div className="template-controls">
          <input
            type="text"
            placeholder="Search templates..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-bar"
          />
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="filter-dropdown"
          >
            <option value="all">All</option>
            <option value="whatsapp">WhatsApp</option>
            <option value="email">Email</option>
          </select>
          <button onClick={() => setIsModalOpen(true)} className="create-template-btn">
            + Create Template
          </button>
        </div>
      </header>

      <main className="template-grid">
        {filteredTemplates.map((template, index) => (
          <div className="template-card" key={index}>
            <h2>{template.title}</h2>
            <p>{template.description}</p>
            <div className="template-actions">
              <button onClick={() => handleEditTemplate(index)} className="edit-btn">
                <FaEdit />
              </button>
              <button onClick={() => handleDeleteTemplate(index)} className="delete-btn">
                <FaTrash />
              </button>
            </div>
          </div>
        ))}
      </main>

      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>{editTemplateIndex !== null ? 'Edit Template' : 'Create New Template'}</h2>
            <input
              type="text"
              placeholder="Title"
              value={newTemplate.title}
              onChange={(e) => setNewTemplate({ ...newTemplate, title: e.target.value })}
              className="modal-input"
            />
            <textarea
              placeholder="Description"
              value={newTemplate.description}
              onChange={(e) => setNewTemplate({ ...newTemplate, description: e.target.value })}
              className="modal-textarea"
            ></textarea>
            <select
              value={newTemplate.type}
              onChange={(e) => setNewTemplate({ ...newTemplate, type: e.target.value })}
              className="modal-select"
            >
              <option value="whatsapp">WhatsApp</option>
              <option value="email">Email</option>
            </select>
            <div className="modal-actions">
              <button onClick={handleCreateOrEditTemplate} className="modal-btn save">Save</button>
              <button onClick={() => setIsModalOpen(false)} className="modal-btn cancel">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
