import { describe, it, expect } from 'vitest';
import { rootAgent } from '../customer_service/agent';

describe('Customer Service Agent', () => {
  it('initializes rootAgent successfully', () => {
    expect(rootAgent).toBeDefined();
    expect(rootAgent.name).toBe('customer_service_agent');
  });
});
