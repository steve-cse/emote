
import { ThemeProvider } from './components/theme-provider';
import Emote from './features/Emote';

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
     <Emote/>
    </ThemeProvider>
  );
}

export default App; 