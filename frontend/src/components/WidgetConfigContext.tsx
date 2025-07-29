import React, { createContext, useContext, useState } from 'react';

export type WidgetPosition = 'bottomRight' | 'bottomLeft' | 'topRight' | 'topLeft';
export type WidgetFont = 'Inter' | 'Roboto' | 'Montserrat' | 'Lato' | 'Poppins' | 'Open Sans' | 'Nunito' | 'Oswald' | 'Raleway' | 'Merriweather';

export interface WidgetConfig {
  enabled: boolean;
  widgetColor: string;
  widgetName: string;
  widgetFont: WidgetFont;
  widgetPosition: WidgetPosition;
  profileMascot: string | null; // base64 or url
}

const defaultConfig: WidgetConfig = {
  enabled: false,
  widgetColor: '#6A5ACD',
  widgetName: 'AI Assistant',
  widgetFont: 'Inter',
  widgetPosition: 'bottomRight',
  profileMascot: null,
};

const WidgetConfigContext = createContext<{
  config: WidgetConfig;
  setConfig: React.Dispatch<React.SetStateAction<WidgetConfig>>;
} | undefined>(undefined);

export const WidgetConfigProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfig] = useState<WidgetConfig>(defaultConfig);
  return (
    <WidgetConfigContext.Provider value={{ config, setConfig }}>
      {children}
    </WidgetConfigContext.Provider>
  );
};

export function useWidgetConfig() {
  const ctx = useContext(WidgetConfigContext);
  if (!ctx) throw new Error('useWidgetConfig must be used within WidgetConfigProvider');
  return ctx;
}
