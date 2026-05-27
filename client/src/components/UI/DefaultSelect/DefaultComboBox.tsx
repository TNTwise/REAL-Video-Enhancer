import './DefaultComboBox.css'
import { useState } from 'react';
import { Select } from '@chakra-ui/react'

export const DefaultCombobox = ({ items }: { items: string[] }) => {
  const [selectedValue, setSelectedValue] = useState('');

  return (
    <Select 
      placeholder='Select option' 
      value={selectedValue} 
      onChange={(e) => setSelectedValue(e.target.value)}
    >
      {items.map((item) => (
        <option key={item} value={item}>
          {item}
        </option>
      ))}
    </Select>
  );
};
