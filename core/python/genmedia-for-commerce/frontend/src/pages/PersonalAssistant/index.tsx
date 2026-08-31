/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useState, useRef, useEffect } from 'react';
import { marked } from 'marked';
import TopNav from '../../components/TopNav';

interface ProductItem {
  images?: { uri: string }[];
  title: string;
  brand?: string;
  price_info?: { price: number };
}

type Message = {
  id: number;
  sender: 'user' | 'aida';
  text: string;
  products?: ProductItem[];
  isDetailsVisible?: boolean;
};

export default function PersonalAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    { id: Date.now(), sender: 'aida', text: "Welcome! I'm Aida, your personal fashion curator. Whether you're attending a gala, planning a vacation, or just refreshing your wardrobe, describe the look you're going for and I'll assemble the perfect outfits for you." }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const sendMessage = async () => {
    if (!inputValue.trim()) return;
    const userMessage: Message = { id: Date.now(), sender: 'user', text: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    const botMessageId = Date.now() + 1;
    let botText = '';
    let botProducts: ProductItem[] | undefined = undefined;

    setMessages(prev => [...prev, { id: botMessageId, sender: 'aida', text: '' }]);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'test_user',
          session_id: 'test_session',
          message: inputValue,
        }),
      });

      if (!response.body) {
        setIsLoading(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let firstChunkReceived = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (!firstChunkReceived) {
          setIsLoading(false);
          firstChunkReceived = true;
        }

        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf('\n');

        while (boundary !== -1) {
          const line = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 1);

          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              if (data.type === 'products') {
                botProducts = botProducts ? [...botProducts, ...data.content] : data.content;
                setMessages(prev => prev.map(msg => 
                  msg.id === botMessageId ? { ...msg, products: botProducts } : msg
                ));
              } else if (data.type === 'text') {
                botText += data.content;
                setMessages(prev => prev.map(msg => 
                  msg.id === botMessageId ? { ...msg, text: botText } : msg
                ));
              } else if (data.type === 'add_to_cart') {
                console.log('Added to cart:', data.content);
              }
            } catch (e) {
              console.error('Error parsing JSON chunk:', e, line);
            }
          }
          boundary = buffer.indexOf('\n');
        }
      }
    } catch (error) {
      setIsLoading(false);
      console.error('Error:', error);
      setMessages(prev => [...prev, { id: Date.now() + 2, sender: 'aida', text: "Sorry, I'm having trouble connecting. Please try again." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-[100dvh] overflow-hidden bg-gm-bg-light dark:bg-gm-bg flex flex-col">
      <TopNav />
      <main className="flex-1 min-h-0 max-w-[1000px] w-full mx-auto p-4 md:pb-6 md:pt-4 flex flex-col relative">
        {/* Chat Messages Area */}
        <div className="flex-1 min-h-0 overflow-y-auto mb-4 bg-white/50 dark:bg-black/20 backdrop-blur-glass rounded-3xl shadow-sm border border-black/5 dark:border-white/10 p-6 flex flex-col gap-6">
          {messages.map(msg => (
            <div key={msg.id} className={`flex flex-col max-w-[85%] animate-fade-in-up ${msg.sender === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
              {msg.text && (
                <div 
                  className={`px-5 py-4 rounded-3xl text-[15px] leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-img:rounded-2xl prose-img:shadow-sm prose-img:max-h-[140px] prose-img:w-auto prose-img:mx-auto prose-img:object-contain prose-img:my-4 prose-headings:font-display prose-headings:font-semibold prose-headings:tracking-tight prose-headings:mt-6 prose-headings:mb-3 prose-p:my-2 prose-ul:my-2 prose-li:my-0.5 prose-strong:text-gm-accent ${
                    msg.sender === 'user' 
                      ? 'bg-gm-accent text-white rounded-br-sm shadow-sm' 
                      : 'bg-white/80 dark:bg-white/5 backdrop-blur-md text-gm-text-primary-light dark:text-gm-text-primary rounded-bl-sm border border-black/5 dark:border-white/10 shadow-sm'
                  }`}
                  dangerouslySetInnerHTML={{ __html: marked.parse(msg.text) }}
                />
              )}
              
              {/* Product Carousel Area */}
              {msg.products && msg.products.length > 0 && (
                <div className="mt-4 flex overflow-x-auto gap-4 max-w-full pb-4 snap-x hide-scrollbar">
                  {msg.products.map((product, idx) => (
                    <div key={idx} className="snap-start shrink-0 w-[200px] bg-white dark:bg-black/40 border border-black/5 dark:border-white/10 rounded-2xl overflow-hidden hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
                      <div className="h-[140px] bg-black/[0.02] dark:bg-white/[0.02] p-4 flex items-center justify-center relative group">
                        <img 
                          src={product.images?.[0]?.uri || '/placeholder.png'} 
                          alt={product.title} 
                          className="max-h-full object-contain mix-blend-multiply dark:mix-blend-normal group-hover:scale-105 transition-transform duration-500" 
                        />
                        <div className="absolute top-2 left-2 px-2 py-1 bg-red-500/90 text-white text-[10px] font-bold rounded">HOT</div>
                      </div>
                      <div className="p-4">
                        <div className="text-[10px] font-bold tracking-wider text-gm-accent mb-1 uppercase">
                          {product.brand || 'Featured'}
                        </div>
                        <h4 className="font-semibold text-sm line-clamp-2 mb-2 text-gm-text-primary-light dark:text-gm-text-primary leading-snug">
                          {product.title}
                        </h4>
                        <div className="flex items-center gap-1 mb-3 text-xs text-gm-text-secondary-light dark:text-gm-text-secondary">
                          <span className="text-yellow-500">★</span> 4.8 (124)
                        </div>
                        <div className="flex items-center justify-between mt-auto">
                          <div className="text-lg font-display text-gm-text-primary-light dark:text-gm-text-primary">
                            €{product.price_info?.price?.toFixed(0) || '--'}
                          </div>
                          <button className="w-8 h-8 rounded-full bg-black/5 dark:bg-white/10 hover:bg-gm-accent hover:text-white transition-colors flex items-center justify-center">
                            +
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          
          {/* Typing Indicator */}
          {isLoading && (
            <div className="self-start px-5 py-4 rounded-2xl bg-white dark:bg-white/5 rounded-bl-sm flex gap-1.5 items-center h-[46px] border border-black/5 dark:border-white/10 shadow-sm animate-fade-in">
              <div className="w-1.5 h-1.5 bg-gm-accent rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-1.5 h-1.5 bg-gm-accent rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-1.5 h-1.5 bg-gm-accent rounded-full animate-bounce"></div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="bg-white/80 dark:bg-black/40 backdrop-blur-xl rounded-2xl shadow-sm border border-black/5 dark:border-white/10 p-2 flex items-center gap-2 relative z-10">
          <button className="w-10 h-10 flex items-center justify-center rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors text-gm-text-secondary-light dark:text-gm-text-secondary">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
            </svg>
          </button>
          <input 
            type="text" 
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
            placeholder="Ask Aida: E.g., 'I need a chic, premium outfit for a summer evening gala...'"
            className="flex-1 bg-transparent border-none outline-none px-2 py-3 text-[15px] text-gm-text-primary-light dark:text-gm-text-primary placeholder:text-black/30 dark:placeholder:text-white/30"
          />
          <button 
            onClick={sendMessage}
            disabled={!inputValue.trim()}
            className="w-10 h-10 flex items-center justify-center bg-gm-accent text-white rounded-xl hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="translate-x-[-1px] translate-y-[1px]">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </main>
    </div>
  );
}
