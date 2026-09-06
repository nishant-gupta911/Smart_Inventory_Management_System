// ── Smart Inventory API Service ──────────────────────────────────────────────
// When VITE_API_URL is set (production on Vercel), all calls go to the
// FastAPI backend on Render. When empty (local dev), uses mock data.

const API_BASE = import.meta.env.VITE_API_URL || '';
const USE_BACKEND = Boolean(API_BASE);

// ─── Generic fetcher ─────────────────────────────────────────────────────────
async function apiFetch(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const url = `${API_BASE}${path}${qs ? '?' + qs : ''}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error ${res.status}: ${url}`);
  return res.json();
}

// ─── Build filter query params (matches backend Query params) ─────────────────
function filterParams(filters = {}) {
  return {
    store:        filters.store        || 'All',
    stock_level:  filters.stockLevel   || 'All',
    expiry_risk:  filters.expiryRisk   || 'All',
    action:       filters.action       || 'All',
  };
}

// ═════════════════════════════════════════════════════════════════════════════
//  MOCK DATA  (mirrors generate_data() in backend/app.py — used locally)
// ═════════════════════════════════════════════════════════════════════════════
const CATEGORIES  = ['DAIRY', 'PRODUCE', 'MEATS', 'SNACKS', 'BAKERY', 'BEVERAGES', 'HOUSEHOLD'];
const STORES      = ['1','2','3','4','5','6','7','8','9','10'];
const CITIES      = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Hyderabad'];
const NGOS        = ['Akshaya Patra', 'Robin Hood Army', 'Feeding India', 'No Food Waste'];
const PRODUCTS    = [
  'Organic Bananas', 'Whole Milk 2%', 'Sliced Bread', 'Greek Yogurt', 'Baby Spinach',
  'Orange Juice', 'Cheddar Block', 'Chicken Breast', 'Brown Rice 5kg', 'Cooking Oil',
  'Paper Towels', 'Cereal Box', 'Bottled Water', 'Potato Chips', 'Frozen Peas',
  'Almond Milk', 'Butter 500g', 'Eggs 12pk', 'Pasta 500g', 'Tomato Sauce',
  'Laundry Detergent', 'Shampoo 400ml', 'Instant Noodles', 'Coffee 250g', 'Orange 1kg',
  'Apple Juice', 'Strawberry Jam', 'Peanut Butter', 'Canned Tuna', 'Olive Oil',
];

function _s(i, o) { const x = Math.sin(i * 37 + o + 1) * 10000; return x - Math.floor(x); }
function _pick(arr, i, o) { return arr[Math.floor(_s(i, o) * arr.length)]; }

function generateMockItems(n = 300) {
  const items = [];
  for (let i = 0; i < n; i++) {
    const stock    = Math.floor(_s(i, 6) * 200);
    const daysExp  = Math.floor(_s(i, 7) * 31);
    const price    = Math.round((_s(i, 8) * 49.5 + 0.5) * 100) / 100;
    const salesAvg = Math.round(_s(i, 9) * 20 * 10) / 10;
    const city     = _pick(CITIES, i, 4);
    const ngo      = _pick(NGOS, i, 5);

    const stockLevel  = stock < 20 ? 'Low' : stock > 120 ? 'High' : 'Normal';
    const expiryRisk  = daysExp === 0 ? 'Expired' : daysExp <= 3 ? 'Near Expiry' : 'Safe';

    let action = 'No Action', discount = 0;
    if (expiryRisk === 'Expired') { action = 'Remove'; }
    else if (expiryRisk === 'Near Expiry') {
      action   = _s(i, 10) > 0.4 ? 'Donate' : 'Apply Discount';
      discount = action === 'Apply Discount' ? [20, 30, 40][Math.floor(_s(i, 11) * 3)] : 0;
    } else if (stockLevel === 'Low') { action = 'Restock'; }
    else if (stockLevel === 'High') { action = 'Apply Discount'; discount = [10, 15, 20][Math.floor(_s(i, 12) * 3)]; }

    const donEligible = action === 'Donate';
    const donStatus   = donEligible ? ['Pending', 'Pending', 'Donated', 'Rejected'][Math.floor(_s(i, 13) * 4)] : 'N/A';

    items.push({
      item_id: `ITEM_${String(i).padStart(5, '0')}`,
      product_name: _pick(PRODUCTS, i, 3),
      store_nbr: _pick(STORES, i, 1),
      current_stock: stock,
      rolling_avg_sales_7: salesAvg,
      days_to_expiry: daysExp,
      unit_price: price,
      Stock_Level: stockLevel,
      Expiry_Risk: expiryRisk,
      Suggested_Discount: discount,
      Reorder: stockLevel === 'Low' ? 'Yes' : 'No',
      Action: action,
      donation_eligible: donEligible,
      donation_status: donStatus,
      city,
      category: _pick(CATEGORIES, i, 2),
      nearest_ngo: ngo,
      ngo_contact: '+91-9999999999',
      ngo_address: `${city} Central`,
      store_latitude: 8.0  + _s(i, 14) * 27,
      store_longitude: 68.0 + _s(i, 15) * 29,
    });
  }
  return items;
}

const MOCK_ITEMS = generateMockItems(300);

function mockFilter(filters = {}) {
  return MOCK_ITEMS.filter(r =>
    (filters.store      === 'All' || !filters.store      || r.store_nbr   === filters.store)      &&
    (filters.stockLevel === 'All' || !filters.stockLevel || r.Stock_Level === filters.stockLevel) &&
    (filters.expiryRisk === 'All' || !filters.expiryRisk || r.Expiry_Risk === filters.expiryRisk) &&
    (filters.action     === 'All' || !filters.action     || r.Action      === filters.action)
  );
}

// ═════════════════════════════════════════════════════════════════════════════
//  PUBLIC API FUNCTIONS  (called by React components)
// ═════════════════════════════════════════════════════════════════════════════

export async function fetchKPIs(filters) {
  if (USE_BACKEND) return apiFetch('/api/kpis', filterParams(filters));
  const items = mockFilter(filters);
  const discItems = items.filter(r => r.Suggested_Discount > 0);
  return {
    total:       items.length,
    total_value: items.reduce((s, r) => s + r.current_stock * r.unit_price, 0),
    high_risk:   items.filter(r => ['Remove','Apply Discount'].includes(r.Action)).length,
    reorder:     items.filter(r => r.Reorder === 'Yes').length,
    overstock:   items.filter(r => r.Stock_Level === 'High').length,
    understock:  items.filter(r => r.Stock_Level === 'Low').length,
    near_expiry: items.filter(r => r.Expiry_Risk === 'Near Expiry').length,
    avg_discount: discItems.length ? discItems.reduce((s,r) => s + r.Suggested_Discount, 0) / discItems.length : 0,
  };
}

export async function fetchFinancial(filters) {
  if (USE_BACKEND) return apiFetch('/api/financial', filterParams(filters));
  const items = mockFilter(filters);
  const disc   = items.filter(r => r.Action === 'Apply Discount');
  const don    = items.filter(r => r.donation_status === 'Donated');
  const rem    = items.filter(r => r.Action === 'Remove');
  const discRev = disc.reduce((s, r) => s + r.unit_price * r.current_stock * (1 - r.Suggested_Discount / 100), 0);
  const donVal  = don.reduce((s, r) => s + r.unit_price * r.current_stock, 0);
  const remVal  = rem.reduce((s, r) => s + r.unit_price * r.current_stock, 0);
  const net     = discRev + donVal - remVal;
  const total   = discRev + donVal + remVal;
  return {
    disc_revenue: discRev, donated_val: donVal, removed_val: remVal,
    net_impact: net, recovery_rate: total > 0 ? (discRev + donVal) / total * 100 : 0,
    items_processed: disc.length + don.length + rem.length,
  };
}

export async function fetchCharts(filters) {
  if (USE_BACKEND) return apiFetch('/api/charts', filterParams(filters));
  const items = mockFilter(filters);
  const countBy = field => {
    const m = {}; items.forEach(r => { m[r[field]] = (m[r[field]] || 0) + 1; });
    return Object.entries(m).map(([name, value]) => ({ name, value }));
  };
  const valByAction = {};
  items.forEach(r => { valByAction[r.Action] = (valByAction[r.Action] || 0) + r.current_stock * r.unit_price; });
  const catAgg = {};
  items.forEach(r => {
    if (!catAgg[r.category]) catAgg[r.category] = { risk: 0, count: 0 };
    if (r.Expiry_Risk !== 'Safe') catAgg[r.category].risk++;
    catAgg[r.category].count++;
  });
  return {
    stock_dist:      countBy('Stock_Level'),
    expiry_dist:     countBy('Expiry_Risk'),
    action_dist:     countBy('Action').sort((a,b) => b.value - a.value),
    value_by_action: Object.entries(valByAction).map(([name, value]) => ({ name, value: Math.round(value) })).sort((a,b)=>b.value-a.value),
    category_risk:   Object.entries(catAgg).map(([cat, d]) => ({ category: cat, riskCount: d.risk, total: d.count, riskPct: Math.round(d.risk/d.count*100) })),
    demand_forecast: [
      { week: 'W1', actual: 1200, predicted: 1180 },
      { week: 'W2', actual: 1350, predicted: 1320 },
      { week: 'W3', actual: 1100, predicted: 1150 },
      { week: 'W4', actual: 1480, predicted: 1450 },
      { week: 'W5', actual: 1600, predicted: 1580 },
      { week: 'W6', actual: 1750, predicted: 1700 },
      { week: 'W7', actual: null, predicted: 1820 },
      { week: 'W8', actual: null, predicted: 1950 },
    ],
  };
}

export async function fetchUrgent(filters, limit = 100) {
  if (USE_BACKEND) return apiFetch('/api/items/urgent', { ...filterParams(filters), limit });
  const items = mockFilter(filters).filter(r => ['Remove','Apply Discount','Restock'].includes(r.Action)).sort((a,b)=>a.days_to_expiry-b.days_to_expiry);
  return { total: items.length, items: items.slice(0, limit) };
}

export async function fetchDiscounts(filters, limit = 100) {
  if (USE_BACKEND) return apiFetch('/api/items/discounts', { ...filterParams(filters), limit });
  const items = mockFilter(filters).filter(r => r.Action === 'Apply Discount').sort((a,b) => b.Suggested_Discount - a.Suggested_Discount)
    .map(r => ({ ...r, original_value: +(r.current_stock * r.unit_price).toFixed(2), discount_amount: +(r.current_stock * r.unit_price * r.Suggested_Discount / 100).toFixed(2), revenue_after: +(r.current_stock * r.unit_price * (1 - r.Suggested_Discount / 100)).toFixed(2) }));
  const sliced = items.slice(0, limit);
  return { total: items.length, items: sliced, totals: { original: sliced.reduce((s,r)=>s+r.original_value,0), discount: sliced.reduce((s,r)=>s+r.discount_amount,0), revenue: sliced.reduce((s,r)=>s+r.revenue_after,0) } };
}

export async function fetchRestock(filters, limit = 100) {
  if (USE_BACKEND) return apiFetch('/api/items/restock', { ...filterParams(filters), limit });
  const items = mockFilter(filters).filter(r => r.Reorder === 'Yes').sort((a,b)=>a.current_stock-b.current_stock);
  return { total: items.length, items: items.slice(0, limit) };
}

export async function fetchRemove(filters, limit = 100) {
  if (USE_BACKEND) return apiFetch('/api/items/remove', { ...filterParams(filters), limit });
  const items = mockFilter(filters).filter(r => r.Action === 'Remove').sort((a,b)=>a.days_to_expiry-b.days_to_expiry).map(r => ({ ...r, loss_value: +(r.current_stock * r.unit_price).toFixed(2) }));
  return { total: items.length, items: items.slice(0, limit), total_loss: items.reduce((s,r)=>s+r.loss_value,0) };
}

export async function fetchDonations(filters, donCity = 'All', donCategory = 'All', limit = 100) {
  if (USE_BACKEND) return apiFetch('/api/items/donations', { ...filterParams(filters), don_city: donCity, don_category: donCategory, limit });
  const items = mockFilter(filters).filter(r => r.donation_eligible && r.Action === 'Donate');
  const fil = items.filter(r => (donCity === 'All' || r.city === donCity) && (donCategory === 'All' || r.category === donCategory));
  const statusCounts = {}; items.forEach(r => { statusCounts[r.donation_status] = (statusCounts[r.donation_status]||0)+1; });
  const cityC = {}, ngoC = {};
  items.forEach(r => { cityC[r.city]=(cityC[r.city]||0)+1; ngoC[r.nearest_ngo]=(ngoC[r.nearest_ngo]||0)+1; });
  return {
    total: items.length,
    status_counts: statusCounts,
    top_cities: Object.entries(cityC).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([city,count])=>({city,count})),
    top_ngos:   Object.entries(ngoC).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([ngo,count])=>({ngo,count})),
    items: fil.slice(0, limit),
  };
}

export function getExportURL(filters) {
  if (!USE_BACKEND) return null;
  const qs = new URLSearchParams(filterParams(filters)).toString();
  return `${API_BASE}/api/items/donations/export?${qs}`;
}

export function getUniqueValues(field) {
  const vals = [...new Set(MOCK_ITEMS.map(r => r[field]).filter(Boolean))].sort();
  return ['All', ...vals];
}
