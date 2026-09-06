import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import {
  Package, AlertTriangle, TrendingUp, Heart, Zap, RefreshCw,
  ChevronRight, CheckCircle, BarChart2, ShoppingCart, DollarSign,
  Trash2, Gift, Download, Filter, X, ChevronDown, Info,
  MapPin, Phone, Tag, TrendingDown, Activity
} from 'lucide-react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import L from 'leaflet';
// Fix Leaflet default icon paths broken by bundlers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});
import {
  fetchKPIs, fetchFinancial, fetchCharts, fetchUrgent,
  fetchDiscounts, fetchRestock, fetchRemove, fetchDonations,
  getExportURL, getUniqueValues
} from './services/api';
import './index.css';

/* ═══════════════════════════════════════════════════════════════
   SLUSH COLOR PALETTE
═══════════════════════════════════════════════════════════════ */
const C = {
  carbon:   '#000000',
  white:    '#ffffff',
  sky:      '#dceeff',
  gray:     '#cccccc',
  mist:     '#e9e9e9',
  blue:     '#4da2ff',
  mint:     '#55db9c',
  lavender: '#e9ccff',
  ember:    '#fb4903',
  sunburst: '#ffd731',
  violet:   '#5c4ade',
};

const PIE_STOCK = { High: C.ember, Normal: C.mint, Low: C.sunburst };
const PIE_EXPIRY = { Safe: C.mint, 'Near Expiry': C.sunburst, Expired: C.ember };
const ACTION_COLORS = {
  'No Action':      C.mist,
  'Apply Discount': C.sunburst,
  'Restock':        C.blue,
  'Remove':         C.ember,
  'Donate':         C.mint,
};
const DONATE_STATUS_COLORS = { Pending: C.sunburst, Donated: C.mint, Rejected: C.ember };

/* ═══════════════════════════════════════════════════════════════
   UTILITY COMPONENTS
═══════════════════════════════════════════════════════════════ */
function fmt(n) {
  if (n === undefined || n === null || isNaN(n)) return '—';
  if (n >= 10_000_000) return `₹${(n / 10_000_000).toFixed(1)}Cr`;
  if (n >= 100_000)    return `₹${(n / 100_000).toFixed(1)}L`;
  return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}

function Metric({ label, value, sub, color, icon: Icon, delta, deltaPositive }) {
  return (
    <div className="kpi-card" style={{ background: color || C.white }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span className="kpi-label" style={{ color: color === C.carbon ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.5)' }}>
          {label}
        </span>
        {Icon && (
          <div style={{ width: 32, height: 32, borderRadius: 1600, border: `1px solid ${color === C.carbon ? 'rgba(255,255,255,0.25)' : C.carbon}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon size={14} color={color === C.carbon ? '#fff' : '#000'} />
          </div>
        )}
      </div>
      <span className="kpi-number" style={{ color: color === C.carbon || color === C.ember || color === C.violet || color === C.blue ? '#fff' : '#000' }}>
        {value}
      </span>
      {delta !== undefined && (
        <span style={{ fontSize: 12, fontWeight: 700, color: (deltaPositive ? C.mint : C.ember) }}>
          {deltaPositive ? '▲' : '▼'} {delta}
        </span>
      )}
      {sub && <span style={{ fontSize: 12, opacity: 0.55, marginTop: 2 }}>{sub}</span>}
    </div>
  );
}

function Badge({ children, variant = 'mist' }) {
  const map = {
    ember: { bg: C.ember, color: '#fff' },
    sunburst: { bg: C.sunburst, color: '#000' },
    mint: { bg: C.mint, color: '#000' },
    lavender: { bg: C.lavender, color: '#000' },
    violet: { bg: C.violet, color: '#fff' },
    mist: { bg: C.mist, color: '#000' },
    blue: { bg: C.blue, color: '#fff' },
    carbon: { bg: C.carbon, color: '#fff' },
  };
  const s = map[variant] || map.mist;
  return (
    <span style={{
      display: 'inline-block', border: `1px solid ${C.carbon}`,
      borderRadius: 1600, padding: '2px 9px',
      fontSize: 11, fontWeight: 700, letterSpacing: '0.032em',
      textTransform: 'uppercase', background: s.bg, color: s.color,
      whiteSpace: 'nowrap',
    }}>{children}</span>
  );
}

function ExpiryBadge({ risk }) {
  const v = risk === 'Expired' ? 'ember' : risk === 'Near Expiry' ? 'sunburst' : 'mint';
  return <Badge variant={v}>{risk}</Badge>;
}
function StockBadge({ level }) {
  const v = level === 'High' ? 'ember' : level === 'Low' ? 'sunburst' : 'mint';
  return <Badge variant={v}>{level}</Badge>;
}
function ActionBadge({ action }) {
  const map = { 'Remove': 'ember', 'Apply Discount': 'sunburst', 'Restock': 'blue', 'Donate': 'mint', 'No Action': 'mist' };
  return <Badge variant={map[action] || 'mist'}>{action}</Badge>;
}
function DonationBadge({ status }) {
  const map = { Pending: 'sunburst', Donated: 'mint', Rejected: 'ember' };
  return <Badge variant={map[status] || 'mist'}>{status}</Badge>;
}

function RiskBar({ value, max = 100 }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct >= 75 ? C.ember : pct >= 40 ? C.sunburst : C.mint;
  return (
    <div style={{ width: '100%', height: 6, background: C.mist, borderRadius: 1600, border: `1px solid ${C.carbon}`, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 1600, transition: 'width 0.5s ease' }} />
    </div>
  );
}

function Select({ label, options, value, onChange, id }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {label && <label style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.032em', textTransform: 'uppercase', opacity: 0.5 }}>{label}</label>}
      <select
        id={id}
        value={value}
        onChange={e => onChange(e.target.value)}
        style={{
          borderRadius: 1600, border: `1px solid ${C.carbon}`, background: C.white,
          fontFamily: 'inherit', fontSize: 13, fontWeight: 700, padding: '8px 14px',
          cursor: 'pointer', appearance: 'none', outline: 'none',
          letterSpacing: '0.02em',
        }}
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

// Table helpers
function Th({ children, style }) {
  return (
    <th style={{
      padding: '14px 16px', fontSize: 11, fontWeight: 700,
      letterSpacing: '0.032em', textTransform: 'uppercase',
      textAlign: 'left', background: C.mist, borderBottom: `1px solid ${C.carbon}`,
      whiteSpace: 'nowrap', ...style
    }}>{children}</th>
  );
}
function Td({ children, style }) {
  return (
    <td style={{
      padding: '12px 16px', fontSize: 13, fontWeight: 500,
      borderBottom: `1px solid ${C.mist}`, verticalAlign: 'middle',
      ...style
    }}>{children}</td>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MARQUEE
═══════════════════════════════════════════════════════════════ */
function MarqueeBand() {
  const items = ['Smart Inventory','·','AI Demand Forecasting','·','Expiry Risk Detection','·',
    'Smart Restocking','·','Donation Management','·','Waste Reduction 28%','·','ML Powered','·'];
  const doubled = [...items, ...items];
  return (
    <div className="marquee-band">
      <div className="marquee-track">
        {doubled.map((t, i) =>
          t === '·'
            ? <span key={i} className="marquee-dot">·</span>
            : <span key={i} className="marquee-item">{t}</span>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   SVG RIBBON
═══════════════════════════════════════════════════════════════ */
function Ribbon({ style }) {
  return (
    <svg viewBox="0 0 900 260" xmlns="http://www.w3.org/2000/svg"
      style={{ position: 'absolute', pointerEvents: 'none', zIndex: 1, ...style }} aria-hidden="true">
      <defs>
        <radialGradient id="rg" cx="50%" cy="30%" r="60%">
          <stop offset="0%"   stopColor="#89c8ff" />
          <stop offset="45%"  stopColor="#4da2ff" />
          <stop offset="100%" stopColor="#1c6bd4" />
        </radialGradient>
      </defs>
      <ellipse cx="450" cy="210" rx="430" ry="40" fill="rgba(0,0,0,0.06)" />
      <path d="M -20 160 C 120 60, 300 220, 450 130 S 720 20, 920 110"
        stroke="url(#rg)" strokeWidth="72" fill="none" strokeLinecap="round" />
      <path d="M -20 148 C 120 46, 300 206, 450 117 S 720 8, 920 98"
        stroke="rgba(255,255,255,0.35)" strokeWidth="18" fill="none" strokeLinecap="round" />
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════════
   FLOATING STICKER
═══════════════════════════════════════════════════════════════ */
function Sticker({ children, color, style }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      width: 72, height: 72, borderRadius: 20, border: `1px solid ${C.carbon}`,
      background: color, position: 'absolute', pointerEvents: 'none', zIndex: 20, ...style,
    }}>{children}</div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   NAV
═══════════════════════════════════════════════════════════════ */
function Nav({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'urgent',    label: '🔴 Urgent' },
    { id: 'discounts', label: '💸 Discounts' },
    { id: 'restock',   label: '📦 Restock' },
    { id: 'remove',    label: '🗑 Remove' },
    { id: 'donations', label: '🤝 Donations' },
  ];
  return (
    <nav className="nav" style={{ gap: 12, flexWrap: 'wrap' }}>
      <div className="nav-logo">S</div>
      <div className="nav-pills" style={{ flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id}
            className={`nav-pill ${activeTab === t.id ? 'nav-pill--active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >{t.label}</button>
        ))}
      </div>
      <button className="btn-cta">
        <RefreshCw size={12} style={{ display: 'inline', marginRight: 6 }} />
        Run ML Pipeline
      </button>
    </nav>
  );
}

/* ═══════════════════════════════════════════════════════════════
   FILTER SIDEBAR / PANEL
═══════════════════════════════════════════════════════════════ */
function FilterBar({ filters, setFilters, itemCount }) {
  const storeOpts  = getUniqueValues('store_nbr');
  const stockOpts  = getUniqueValues('Stock_Level');
  const expiryOpts = getUniqueValues('Expiry_Risk');
  const actionOpts = getUniqueValues('Action');

  const active = Object.values(filters).filter(v => v !== 'All').length;

  return (
    <div style={{
      background: C.mist, border: `1px solid ${C.carbon}`, borderRadius: 20,
      padding: '16px 24px', display: 'flex', gap: 16, flexWrap: 'wrap',
      alignItems: 'flex-end', marginBottom: 32,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <Filter size={16} />
        <span style={{ fontSize: 13, fontWeight: 700 }}>Filters</span>
        <Badge variant={active > 0 ? 'violet' : 'mist'}>{itemCount.toLocaleString()} items</Badge>
      </div>
      <Select id="store-filter"  label="Store"        options={storeOpts}  value={filters.store}      onChange={v => setFilters(f => ({...f, store: v}))} />
      <Select id="stock-filter"  label="Stock Level"  options={stockOpts}  value={filters.stockLevel} onChange={v => setFilters(f => ({...f, stockLevel: v}))} />
      <Select id="expiry-filter" label="Expiry Risk"  options={expiryOpts} value={filters.expiryRisk} onChange={v => setFilters(f => ({...f, expiryRisk: v}))} />
      <Select id="action-filter" label="Action"       options={actionOpts} value={filters.action}     onChange={v => setFilters(f => ({...f, action: v}))} />
      {active > 0 && (
        <button className="btn-ghost" style={{ alignSelf: 'flex-end' }}
          onClick={() => setFilters({ store:'All', stockLevel:'All', expiryRisk:'All', action:'All' })}>
          <X size={12} style={{ marginRight: 4, display: 'inline' }} />
          Clear All
        </button>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   DASHBOARD VIEW — all charts + financial + KPIs
═══════════════════════════════════════════════════════════════ */
function DashboardView({ fin, kpis, charts, loading }) {
  const stockDist  = charts?.stock_dist      || [];
  const expiryDist = charts?.expiry_dist     || [];
  const actionDist = charts?.action_dist     || [];
  const valueByAct = charts?.value_by_action || [];
  const catData    = charts?.category_risk   || [];
  const demand     = charts?.demand_forecast || [];

  if (loading || !fin || !kpis) return <LoadingBand />;

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{ background: C.white, border: `1px solid ${C.carbon}`, borderRadius: 16, padding: '10px 14px', fontFamily: 'inherit', fontSize: 13 }}>
        <strong>{label}</strong>
        {payload.map(p => <div key={p.name}><span style={{ color: p.color }}>■</span> {p.name}: <strong>{p.value?.toLocaleString()}</strong></div>)}
      </div>
    );
  };

  const PieTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{ background: C.white, border: `1px solid ${C.carbon}`, borderRadius: 16, padding: '10px 14px', fontSize: 13 }}>
        <strong>{payload[0].name}</strong>: {payload[0].value}
      </div>
    );
  };

  return (
    <>
      {/* ── HERO ── */}
      <section className="section-band section-band--sky" style={{ minHeight: '80vh' }}>
        <Ribbon style={{ bottom: -40, left: -60, width: 900, opacity: 0.85 }} />
        <Sticker color={C.ember}    style={{ top: '16%', left: '6%',   transform: 'rotate(-14deg)' }}><Package    size={34} color="#fff" /></Sticker>
        <Sticker color={C.sunburst} style={{ top: '10%', right: '8%',  transform: 'rotate(10deg)'  }}><TrendingUp size={34} color="#000" /></Sticker>
        <Sticker color={C.mint}     style={{ bottom: '20%', right: '10%', transform: 'rotate(-8deg)' }}><CheckCircle size={34} color="#000" /></Sticker>
        <Sticker color={C.violet}   style={{ bottom: '15%', left: '5%', transform: 'rotate(12deg)' }}><Zap         size={34} color="#fff" /></Sticker>

        <div className="content-wrap" style={{ textAlign: 'center' }}>
          <span className="label" style={{ opacity: 0.45, display: 'block', marginBottom: 16 }}>Smart Inventory Management System — Sparkathon 2025</span>
          <h1 className="display display--lg" style={{ fontSize: 'clamp(72px, 12vw, 180px)', lineHeight: 0.80 }}>
            INVENTORY<br/>INTELLIGENCE
          </h1>
          <p className="subheading" style={{ marginTop: 28, maxWidth: 560, margin: '28px auto 48px', opacity: 0.7 }}>
            Real-time ML predictions for demand forecasting, expiry risk detection &amp; smart restocking — at Walmart scale.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 12 }}>
            <button className="btn-cta" style={{ fontSize: 14, padding: '12px 32px' }}>View Dashboard</button>
            <button className="btn-ghost" style={{ fontSize: 14, padding: '12px 32px' }}>Download Report</button>
          </div>
        </div>
      </section>

      {/* ── FINANCIAL IMPACT SUMMARY (app.py lines 176-203) ── */}
      <section className="section-band section-band--paper" style={{ minHeight: 'auto', padding: '64px 48px' }}>
        <div className="content-wrap">
          <div className="section-header">
            <div>
              <h2 className="display display--sm" style={{ marginBottom: 8 }}>FINANCIAL IMPACT</h2>
              <p className="body-lg" style={{ opacity: 0.5 }}>Revenue from discounts · Donated goods value · Loss from removals</p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
            <Metric label="Revenue from Discounts" value={fmt(fin.disc_revenue)}  color={C.mint}     icon={DollarSign} />
            <Metric label="Donated Goods Value"    value={fmt(fin.donated_val)}   color={C.violet}   icon={Gift} />
            <Metric label="Loss from Removals"     value={fmt(fin.removed_val)}   color={C.ember}    icon={Trash2} />
            <Metric label="Net Financial Impact"   value={fmt(fin.net_impact)}    color={fin.net_impact >= 0 ? C.mint : C.ember} icon={TrendingUp} />
          </div>

          {(fin.disc_revenue + fin.donated_val + fin.removed_val) > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <Metric label="Value Recovery Rate"  value={`${fin.recovery_rate.toFixed(1)}%`} color={C.sky} icon={TrendingUp} />
              <Metric label="Items Processed"      value={fin.items_processed.toLocaleString()} color={C.lavender} icon={Package} />
            </div>
          )}
        </div>
      </section>

      {/* ── KPI GRID (app.py lines 206-221) ── */}
      <section className="section-band section-band--gray" style={{ minHeight: 'auto', padding: '64px 48px' }}>
        <div className="content-wrap">
          <div className="section-header">
            <div>
              <h2 className="display display--sm" style={{ marginBottom: 8 }}>KEY METRICS</h2>
              <p className="body-lg" style={{ opacity: 0.5 }}>Live KPIs from your ML pipeline output</p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
            <Metric label="Total Items"      value={kpis.total.toLocaleString()}       color={C.white}    icon={Package} />
            <Metric label="Inventory Value"  value={fmt(kpis.total_value)}             color={C.blue}     icon={DollarSign} />
            <Metric label="High Risk Items"  value={kpis.high_risk.toLocaleString()}   color={C.ember}    icon={AlertTriangle} />
            <Metric label="Reorder Needed"   value={kpis.reorder.toLocaleString()}     color={C.sunburst} icon={ShoppingCart} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            <Metric label="Overstocked"      value={kpis.overstock.toLocaleString()}   color={C.white}    icon={TrendingUp} />
            <Metric label="Understocked"     value={kpis.understock.toLocaleString()}  color={C.lavender} icon={TrendingDown} />
            <Metric label="Near Expiry"      value={kpis.near_expiry.toLocaleString()} color={C.violet}   icon={AlertTriangle} />
            <Metric label="Avg Discount"     value={`${kpis.avg_discount.toFixed(1)}%`} color={C.white}    icon={Tag} />
          </div>
        </div>
      </section>

      {/* ── CHARTS (app.py lines 224-257) ── */}
      <section className="section-band section-band--sky" style={{ minHeight: 'auto', padding: '64px 48px' }}>
        <div className="content-wrap">
          <div className="section-header">
            <h2 className="display display--sm">ANALYTICS</h2>
          </div>

          {/* Row 1: two pie charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
            <div className="chart-wrap">
              <h3 className="heading-sm" style={{ marginBottom: 20 }}>Stock Level Distribution</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={stockDist} cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine>
                    {stockDist.map((entry, i) => <Cell key={i} fill={PIE_STOCK[entry.name] || C.mist} stroke={C.carbon} strokeWidth={1} />)}
                  </Pie>
                  <Tooltip content={<PieTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-wrap">
              <h3 className="heading-sm" style={{ marginBottom: 20 }}>Expiry Risk Analysis</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={expiryDist} cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine>
                    {expiryDist.map((entry, i) => <Cell key={i} fill={PIE_EXPIRY[entry.name] || C.mist} stroke={C.carbon} strokeWidth={1} />)}
                  </Pie>
                  <Tooltip content={<PieTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Row 2: Action bar */}
          <div className="chart-wrap" style={{ marginBottom: 24 }}>
            <h3 className="heading-sm" style={{ marginBottom: 20 }}>⚡ Action Recommendations</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={actionDist} margin={{ top: 10, right: 20, left: 0, bottom: 0 }} barSize={48}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,0,0,0.08)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontFamily: 'Inter', fontSize: 12, fontWeight: 700 }} />
                <YAxis tick={{ fontFamily: 'Inter', fontSize: 12 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" name="Items" radius={[8, 8, 0, 0]}>
                  {actionDist.map((entry, i) => <Cell key={i} fill={ACTION_COLORS[entry.name] || C.blue} stroke={C.carbon} strokeWidth={1} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Row 3: Value by action */}
          <div className="chart-wrap" style={{ marginBottom: 24 }}>
            <h3 className="heading-sm" style={{ marginBottom: 20 }}>💰 Inventory Value by Action</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={valueByAct} margin={{ top: 10, right: 20, left: 0, bottom: 0 }} barSize={48}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,0,0,0.08)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontFamily: 'Inter', fontSize: 12, fontWeight: 700 }} />
                <YAxis tick={{ fontFamily: 'Inter', fontSize: 12 }} tickFormatter={v => `₹${(v/1000).toFixed(0)}K`} />
                <Tooltip content={<CustomTooltip />} formatter={v => [fmt(v), 'Value']} />
                <Bar dataKey="value" name="Inventory Value" radius={[8, 8, 0, 0]}>
                  {valueByAct.map((entry, i) => <Cell key={i} fill={ACTION_COLORS[entry.name] || C.blue} stroke={C.carbon} strokeWidth={1} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Row 4: Demand forecast + category risk */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 24 }}>
            <div className="chart-wrap">
              <h3 className="heading-sm" style={{ marginBottom: 20 }}>📈 Demand Forecast (8-Week)</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={demand} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,0,0,0.08)" />
                  <XAxis dataKey="week" tick={{ fontFamily: 'Inter', fontSize: 12, fontWeight: 700 }} />
                  <YAxis tick={{ fontFamily: 'Inter', fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontFamily: 'Inter', fontSize: 13, fontWeight: 700 }} />
                  <Line type="monotone" dataKey="actual"    name="Actual"    stroke={C.carbon} strokeWidth={2} dot={{ r: 4 }} connectNulls={false} />
                  <Line type="monotone" dataKey="predicted" name="Predicted" stroke={C.violet} strokeWidth={2} strokeDasharray="6 3" dot={{ r: 4, fill: C.violet }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-wrap">
              <h3 className="heading-sm" style={{ marginBottom: 20 }}>Risk by Category</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {catData.slice(0, 6).map(c => (
                  <div key={c.category}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ fontSize: 13, fontWeight: 700 }}>{c.category}</span>
                      <span style={{ fontSize: 12, opacity: 0.55 }}>{c.riskPct}% risk</span>
                    </div>
                    <RiskBar value={c.riskPct} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Bottom summary strip ── */}
      <section style={{ background: C.carbon, padding: '40px 48px' }}>
        <div style={{ maxWidth: 1440, margin: '0 auto', display: 'flex', gap: 48, flexWrap: 'wrap', alignItems: 'center' }}>
          {[
            { v: kpis.total.toLocaleString(), l: 'SKUs Tracked' },
            { v: `${fin.recovery_rate.toFixed(0)}%`, l: 'Value Recovery' },
            { v: fmt(fin.net_impact), l: 'Net Impact' },
            { v: kpis.reorder.toString(), l: 'Items to Reorder' },
          ].map((s, i) => (
            <React.Fragment key={s.l}>
              {i > 0 && <div style={{ width: 1, height: 48, background: 'rgba(255,255,255,0.15)', flexShrink: 0 }} />}
              <div>
                <div style={{ fontFamily: 'var(--font-lateral)', fontSize: 44, fontWeight: 800, lineHeight: 1, color: '#fff', textTransform: 'uppercase' }}>{s.v}</div>
                <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.032em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.45)', marginTop: 4 }}>{s.l}</div>
              </div>
            </React.Fragment>
          ))}
        </div>
      </section>
    </>
  );
}

/* ─── Loading skeleton ───────────────────────────────────────────────────── */
function LoadingBand() {
  return (
    <div style={{ minHeight: '60vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20, background: 'var(--color-sky-wash)' }}>
      <div style={{ fontFamily: 'var(--font-lateral)', fontSize: 64, fontWeight: 800, textTransform: 'uppercase', lineHeight: 0.8, opacity: 0.15, animation: 'pulse 1.4s ease-in-out infinite' }}>LOADING</div>
      <p style={{ fontSize: 14, fontWeight: 700, opacity: 0.4, letterSpacing: '0.04em', textTransform: 'uppercase' }}>Fetching data...</p>
      <style>{`@keyframes pulse { 0%,100%{opacity:.1} 50%{opacity:.3} }`}</style>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   URGENT TAB (app.py tab1)
═══════════════════════════════════════════════════════════════ */
function UrgentView({ data, loading }) {
  if (loading) return <LoadingBand />;
  const urgent = data?.items || [];
  const LIMIT = 100;

  return (
    <section className="section-band section-band--sky" style={{ minHeight: '100vh', padding: '64px 48px' }}>
      <Ribbon style={{ top: -80, right: -80, width: 700, transform: 'rotate(15deg)', opacity: 0.55 }} />
      <Sticker color={C.ember} style={{ top: 48, right: 48, transform: 'rotate(-8deg)' }}><AlertTriangle size={34} color="#fff" /></Sticker>

      <div className="content-wrap">
        <div className="section-header">
          <div>
            <h1 className="display display--sm" style={{ marginBottom: 8 }}>URGENT<br/>ITEMS</h1>
            <p className="body-lg" style={{ opacity: 0.5 }}>
              <strong>{urgent.length.toLocaleString()}</strong> items requiring immediate attention
              {urgent.length > LIMIT && ` (showing first ${LIMIT})`}
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Badge variant="ember">Remove</Badge>
            <Badge variant="sunburst">Apply Discount</Badge>
            <Badge variant="blue">Restock</Badge>
          </div>
        </div>

        <div className="table-card">
          <table className="data-table" style={{ width: '100%' }}>
            <thead><tr>
              <Th>Item ID</Th><Th>Product</Th><Th>Store</Th>
              <Th>Stock</Th><Th>Stock Level</Th><Th>Expiry Risk</Th>
              <Th>Days Left</Th><Th>Action</Th>
            </tr></thead>
            <tbody>
              {urgent.slice(0, LIMIT).map(r => (
                <tr key={r.item_id}>
                  <Td><span style={{ fontFamily: 'monospace', fontSize: 12, opacity: 0.55 }}>{r.item_id}</span></Td>
                  <Td><strong>{r.product_name}</strong></Td>
                  <Td>#{r.store_nbr}</Td>
                  <Td style={{ fontVariantNumeric: 'tabular-nums' }}>{r.current_stock}</Td>
                  <Td><StockBadge level={r.Stock_Level} /></Td>
                  <Td><ExpiryBadge risk={r.Expiry_Risk} /></Td>
                  <Td>
                    <span style={{
                      background: r.days_to_expiry <= 1 ? C.ember : r.days_to_expiry <= 3 ? C.sunburst : C.mist,
                      color: r.days_to_expiry <= 1 ? '#fff' : '#000',
                      border: `1px solid ${C.carbon}`, borderRadius: 1600, padding: '2px 10px',
                      fontSize: 12, fontWeight: 700,
                    }}>{r.days_to_expiry}d</span>
                  </Td>
                  <Td><ActionBadge action={r.Action} /></Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   DISCOUNTS TAB (app.py tab2)
═══════════════════════════════════════════════════════════════ */
function DiscountsView({ data, loading }) {
  if (loading) return <LoadingBand />;
  const di     = data?.items   || [];
  const totals = data?.totals  || {};
  const LIMIT  = 100;
  const totOrig = totals.original || 0;
  const totDisc = totals.discount || 0;
  const totRev  = totals.revenue  || 0;

  return (
    <section className="section-band section-band--paper" style={{ minHeight: '100vh', padding: '64px 48px' }}>
      <Ribbon style={{ bottom: -60, left: -80, width: 800, opacity: 0.5 }} />
      <Sticker color={C.sunburst} style={{ top: 48, right: 48, transform: 'rotate(10deg)' }}><Tag size={34} color="#000" /></Sticker>

      <div className="content-wrap">
        <div className="section-header">
          <div>
            <h1 className="display display--sm" style={{ marginBottom: 8 }}>DISCOUNT<br/>PLANNER</h1>
            <p className="body-lg" style={{ opacity: 0.5 }}>
              <strong>{di.length.toLocaleString()}</strong> items with discount recommendations
              {di.length > LIMIT && ` (showing first ${LIMIT})`}
            </p>
          </div>
        </div>

        {di.length > 0 && (
          <>
            {/* Summary strip */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 28 }}>
              <div className="card" style={{ background: C.mist, textAlign: 'center' }}>
                <div className="caption" style={{ opacity: 0.55, marginBottom: 4 }}>Original Value</div>
                <div style={{ fontFamily: 'var(--font-lateral)', fontSize: 36, fontWeight: 800, textTransform: 'uppercase' }}>{fmt(totOrig)}</div>
              </div>
              <div className="card" style={{ background: C.sunburst, textAlign: 'center' }}>
                <div className="caption" style={{ opacity: 0.65, marginBottom: 4 }}>Total Discount Applied</div>
                <div style={{ fontFamily: 'var(--font-lateral)', fontSize: 36, fontWeight: 800, textTransform: 'uppercase' }}>{fmt(totDisc)}</div>
              </div>
              <div className="card" style={{ background: C.mint, textAlign: 'center' }}>
                <div className="caption" style={{ opacity: 0.65, marginBottom: 4 }}>Revenue After Discount</div>
                <div style={{ fontFamily: 'var(--font-lateral)', fontSize: 36, fontWeight: 800, textTransform: 'uppercase' }}>{fmt(totRev)}</div>
              </div>
            </div>

            <div className="table-card">
              <table className="data-table">
                <thead><tr>
                  <Th>Product</Th><Th>Store</Th><Th>Stock</Th>
                  <Th>Unit Price</Th><Th>Discount</Th>
                  <Th>Orig. Value</Th><Th>Disc. Amt</Th>
                  <Th>Revenue</Th><Th>Expiry Risk</Th>
                </tr></thead>
                <tbody>
                  {di.slice(0, LIMIT).map(r => (
                    <tr key={r.item_id}>
                      <Td><strong>{r.product_name}</strong></Td>
                      <Td>#{r.store_nbr}</Td>
                      <Td style={{ fontVariantNumeric: 'tabular-nums' }}>{r.current_stock}</Td>
                      <Td style={{ fontVariantNumeric: 'tabular-nums' }}>₹{r.unit_price.toFixed(2)}</Td>
                      <Td>
                        <span style={{ background: C.sunburst, border: `1px solid ${C.carbon}`, borderRadius: 1600, padding: '2px 10px', fontSize: 12, fontWeight: 700 }}>
                          {r.Suggested_Discount}%
                        </span>
                      </Td>
                      <Td style={{ fontVariantNumeric: 'tabular-nums' }}>₹{r.original_value.toLocaleString()}</Td>
                      <Td style={{ fontVariantNumeric: 'tabular-nums', color: C.ember, fontWeight: 700 }}>-₹{r.discount_amount.toLocaleString()}</Td>
                      <Td style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 700, color: '#1a7a1a' }}>₹{r.revenue_after.toLocaleString()}</Td>
                      <Td><ExpiryBadge risk={r.Expiry_Risk} /></Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        {di.length === 0 && (
          <div className="card" style={{ textAlign: 'center', padding: 60, opacity: 0.5 }}>
            <Tag size={48} style={{ marginBottom: 16, opacity: 0.3 }} />
            <p className="subheading">No discount items in current filter.</p>
          </div>
        )}
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   RESTOCK TAB (app.py tab3)
═══════════════════════════════════════════════════════════════ */
function RestockView({ data, loading }) {
  if (loading) return <LoadingBand />;
  const ri = data?.items || [];
  const [approved, setApproved] = useState({});
  const LIMIT = 100;

  return (
    <section className="section-band section-band--gray" style={{ minHeight: '100vh', padding: '64px 48px' }}>
      <Ribbon style={{ bottom: -60, left: -80, width: 800, opacity: 0.5 }} />
      <Sticker color={C.mint} style={{ top: 48, right: 48, transform: 'rotate(10deg)' }}><ShoppingCart size={34} color="#000" /></Sticker>

      <div className="content-wrap">
        <div className="section-header">
          <div>
            <h1 className="display display--sm" style={{ marginBottom: 8 }}>RESTOCK<br/>PLANNER</h1>
            <p className="body-lg" style={{ opacity: 0.5 }}>
              <strong>{ri.length.toLocaleString()}</strong> items needing restock
              {ri.length > LIMIT && ` (showing first ${LIMIT})`}
            </p>
          </div>
          <button className="btn-cta" onClick={() => {
            const all = {};
            ri.forEach(r => { all[r.item_id] = true; });
            setApproved(all);
          }}>
            <CheckCircle size={14} style={{ display: 'inline', marginRight: 6 }} />
            Approve All
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {ri.slice(0, LIMIT).map(r => {
            const needed = Math.max(Math.ceil(r.rolling_avg_sales_7 * 14) - r.current_stock, 0);
            const coverage = r.rolling_avg_sales_7 > 0 ? Math.min((r.current_stock / (r.rolling_avg_sales_7 * 14)) * 100, 100) : 100;
            const isApproved = approved[r.item_id];
            return (
              <div key={r.item_id} className="card"
                style={{
                  display: 'flex', alignItems: 'center', gap: 24,
                  background: isApproved ? C.mint : C.white,
                  transition: 'background 0.2s ease',
                }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
                    <strong style={{ fontSize: 16, fontWeight: 700 }}>{r.product_name}</strong>
                    <Badge variant="mist">{r.category}</Badge>
                    <Badge variant="sunburst">#{r.store_nbr}</Badge>
                  </div>
                  <div style={{ display: 'flex', gap: 32, marginBottom: 10, flexWrap: 'wrap' }}>
                    <div>
                      <div className="caption" style={{ opacity: 0.5, marginBottom: 2 }}>Current Stock</div>
                      <div style={{ fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{r.current_stock}</div>
                    </div>
                    <div>
                      <div className="caption" style={{ opacity: 0.5, marginBottom: 2 }}>7-Day Avg Sales</div>
                      <div style={{ fontWeight: 700 }}>{r.rolling_avg_sales_7.toFixed(1)}/day</div>
                    </div>
                    <div>
                      <div className="caption" style={{ opacity: 0.5, marginBottom: 2 }}>Days Left</div>
                      <div style={{ fontWeight: 700 }}>{r.days_to_expiry}d</div>
                    </div>
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span className="caption" style={{ opacity: 0.45 }}>14-day stock coverage</span>
                      <span className="caption" style={{ fontWeight: 700 }}>{coverage.toFixed(0)}%</span>
                    </div>
                    <RiskBar value={100 - coverage} />
                  </div>
                </div>
                <div style={{ textAlign: 'center', flexShrink: 0, minWidth: 90 }}>
                  <div className="caption" style={{ opacity: 0.5, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.032em', fontWeight: 700 }}>Suggested</div>
                  <div style={{ fontFamily: 'var(--font-lateral)', fontSize: 44, fontWeight: 800, lineHeight: 1, textTransform: 'uppercase' }}>{needed || '—'}</div>
                  <div className="caption" style={{ opacity: 0.45 }}>units</div>
                </div>
                <div style={{ flexShrink: 0 }}>
                  {isApproved ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                      <CheckCircle size={28} color={C.carbon} />
                      <span className="caption" style={{ fontWeight: 700 }}>Approved</span>
                    </div>
                  ) : (
                    <button className="btn-cta" onClick={() => setApproved(p => ({ ...p, [r.item_id]: true }))}>
                      Approve
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {ri.length === 0 && (
          <div className="card" style={{ textAlign: 'center', padding: 60, opacity: 0.5 }}>
            <ShoppingCart size={48} style={{ marginBottom: 16, opacity: 0.3 }} />
            <p className="subheading">No restock items in current filter.</p>
          </div>
        )}
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   REMOVE TAB (app.py tab4)
═══════════════════════════════════════════════════════════════ */
function RemoveView({ data, loading }) {
  if (loading) return <LoadingBand />;
  const rmi       = data?.items      || [];
  const LIMIT     = 100;
  const totalLoss = data?.total_loss || 0;

  return (
    <section className="section-band section-band--paper" style={{ minHeight: '100vh', padding: '64px 48px' }}>
      <div className="content-wrap">
        <div className="section-header">
          <div>
            <h1 className="display display--sm" style={{ marginBottom: 8 }}>REMOVAL<br/>QUEUE</h1>
            <p className="body-lg" style={{ opacity: 0.5 }}>
              <strong>{rmi.length.toLocaleString()}</strong> expired items to remove
            </p>
          </div>
        </div>

        {rmi.length > 0 && (
          <>
            <div className="card" style={{ background: C.ember, color: '#fff', marginBottom: 24, display: 'flex', alignItems: 'center', gap: 16, padding: '20px 28px' }}>
              <Trash2 size={24} />
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.032em', textTransform: 'uppercase', opacity: 0.7 }}>Total Loss from Expired Inventory</div>
                <div style={{ fontFamily: 'var(--font-lateral)', fontSize: 40, fontWeight: 800, lineHeight: 1, textTransform: 'uppercase' }}>{fmt(totalLoss)}</div>
              </div>
            </div>

            <div className="table-card">
              <table className="data-table">
                <thead><tr>
                  <Th>Item ID</Th><Th>Product</Th><Th>Store</Th>
                  <Th>Stock</Th><Th>Days to Expiry</Th>
                  <Th>Unit Price</Th><Th>Loss Value</Th>
                </tr></thead>
                <tbody>
                  {rmi.slice(0, LIMIT).map(r => (
                    <tr key={r.item_id}>
                      <Td><span style={{ fontFamily: 'monospace', fontSize: 12, opacity: 0.55 }}>{r.item_id}</span></Td>
                      <Td><strong>{r.product_name}</strong></Td>
                      <Td>#{r.store_nbr}</Td>
                      <Td style={{ fontVariantNumeric: 'tabular-nums' }}>{r.current_stock}</Td>
                      <Td>
                        <span style={{ background: C.ember, color: '#fff', border: `1px solid ${C.carbon}`, borderRadius: 1600, padding: '2px 10px', fontSize: 12, fontWeight: 700 }}>
                          {r.days_to_expiry}d
                        </span>
                      </Td>
                      <Td style={{ fontVariantNumeric: 'tabular-nums' }}>₹{r.unit_price.toFixed(2)}</Td>
                      <Td style={{ fontWeight: 700, color: C.ember, fontVariantNumeric: 'tabular-nums' }}>₹{r.loss_value.toLocaleString()}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        {rmi.length === 0 && (
          <div className="card" style={{ textAlign: 'center', padding: 60, opacity: 0.5 }}>
            <CheckCircle size={48} style={{ marginBottom: 16, opacity: 0.3 }} />
            <p className="subheading">No items marked for removal in current filter.</p>
          </div>
        )}
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   DONATIONS TAB (app.py tab5 — full Donation Management Center)
═══════════════════════════════════════════════════════════════ */
function DonationsView({ data, loading, donFilters, setDonFilters }) {
  if (loading) return <LoadingBand />;
  const [donationStates, setDonationStates] = useState({});
  const donCityFilter = donFilters?.city || 'All';
  const donCatFilter  = donFilters?.cat  || 'All';
  const setDonCityFilter = (v) => setDonFilters(f => ({ ...f, city: v }));
  const setDonCatFilter  = (v) => setDonFilters(f => ({ ...f, cat: v }));
  const LIMIT = 50;

  const baseItems = data?.items || [];

  // Merge local state changes (mirrors app.py donation_status updates)
  const donItems = useMemo(() =>
    baseItems.map(r => ({ ...r, donation_status: donationStates[r.item_id] || r.donation_status })),
    [baseItems, donationStates]
  );

  const pending   = donItems.filter(r => r.donation_status === 'Pending');
  const donated   = donItems.filter(r => r.donation_status === 'Donated');
  const rejected  = donItems.filter(r => r.donation_status === 'Rejected');

  // City/category filters for donation table
  const cities = ['All', ...new Set(donItems.map(r => r.city))].sort();
  const cats   = ['All', ...new Set(donItems.map(r => r.category))].sort();

  const filtered = donItems.filter(r =>
    (donCityFilter === 'All' || r.city === donCityFilter) &&
    (donCatFilter  === 'All' || r.category === donCatFilter)
  );

  // Bulk actions
  const markAllPending = (status) => {
    const updates = {};
    pending.forEach(r => { updates[r.item_id] = status; });
    setDonationStates(prev => ({ ...prev, ...updates }));
  };

  // Individual action
  const markOne = (item_id, status) => {
    setDonationStates(prev => ({ ...prev, [item_id]: status }));
  };

  // CSV Export
  const exportCSV = () => {
    const rows = [Object.keys(filtered[0] || {}).join(',')];
    filtered.forEach(r => rows.push(Object.values(r).join(',')));
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url;
    a.download = `donations_${new Date().toISOString().slice(0,10)}.csv`; a.click();
  };

  // City / NGO top-5 (mirrors app.py col_city + col_ngo)
  const cityCounts = {};
  const ngoCounts  = {};
  donItems.forEach(r => {
    cityCounts[r.city] = (cityCounts[r.city] || 0) + 1;
    ngoCounts[r.nearest_ngo] = (ngoCounts[r.nearest_ngo] || 0) + 1;
  });
  const top5Cities = Object.entries(cityCounts).sort((a,b)=>b[1]-a[1]).slice(0,5);
  const top5NGOs   = Object.entries(ngoCounts).sort((a,b)=>b[1]-a[1]).slice(0,5);

  const StatusPieData = [
    { name: 'Pending',  value: pending.length  },
    { name: 'Donated',  value: donated.length  },
    { name: 'Rejected', value: rejected.length },
  ].filter(d => d.value > 0);

  if (donItems.length === 0) {
    return (
      <section className="section-band section-band--sky" style={{ padding: '64px 48px' }}>
        <div className="content-wrap" style={{ textAlign: 'center', padding: '120px 0' }}>
          <Gift size={64} style={{ opacity: 0.2, marginBottom: 24 }} />
          <h2 className="heading-sm">No donation-eligible items in current filter.</h2>
        </div>
      </section>
    );
  }

  return (
    <section className="section-band section-band--sky" style={{ minHeight: '100vh', padding: '64px 48px' }}>
      <Ribbon style={{ bottom: -60, right: -80, width: 700, opacity: 0.55 }} />
      <Sticker color={C.mint} style={{ top: 48, right: 48, transform: 'rotate(-10deg)' }}><Gift size={34} color="#000" /></Sticker>

      <div className="content-wrap">
        <div className="section-header">
          <div>
            <h1 className="display display--sm" style={{ marginBottom: 8 }}>DONATION<br/>CENTER</h1>
            <p className="body-lg" style={{ opacity: 0.5 }}>Manage NGO donations for near-expiry items</p>
          </div>
        </div>

        {/* Status KPIs (app.py d1–d4) */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 32 }}>
          <Metric label="🎁 Eligible"   value={donItems.length.toString()}  color={C.lavender} icon={Gift} />
          <Metric label="🟡 Pending"    value={pending.length.toString()}   color={C.sunburst} icon={AlertTriangle} />
          <Metric label="🟢 Donated"    value={donated.length.toString()}   color={C.mint}     icon={CheckCircle} />
          <Metric label="🔴 Rejected"   value={rejected.length.toString()}  color={C.ember}    icon={X} />
        </div>

        {/* Status pie + city/ngo charts (app.py col_city + col_ngo) */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20, marginBottom: 32 }}>
          <div className="chart-wrap">
            <h3 className="heading-sm" style={{ marginBottom: 16 }}>Donation Status</h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={StatusPieData} cx="50%" cy="50%" outerRadius={85} dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine>
                  {StatusPieData.map((entry, i) => (
                    <Cell key={i} fill={DONATE_STATUS_COLORS[entry.name] || C.mist} stroke={C.carbon} strokeWidth={1} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h3 className="heading-sm" style={{ marginBottom: 16 }}>🏙️ Top Cities</h3>
            {top5Cities.map(([city, cnt], i) => (
              <div key={city} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span style={{ fontSize: 14, fontWeight: 500 }}>{i+1}. <strong>{city}</strong></span>
                <Badge variant="lavender">{cnt}</Badge>
              </div>
            ))}
          </div>

          <div className="card">
            <h3 className="heading-sm" style={{ marginBottom: 16 }}>🏢 Top NGOs</h3>
            {top5NGOs.map(([ngo, cnt], i) => (
              <div key={ngo} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span style={{ fontSize: 14, fontWeight: 500 }}>{i+1}. <strong>{ngo}</strong></span>
                <Badge variant="mint">{cnt}</Badge>
              </div>
            ))}
          </div>
        </div>

        {/* City/Category filter (app.py f1, f2) */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <Select id="don-city" label="Filter by City" options={cities} value={donCityFilter} onChange={setDonCityFilter} />
          <Select id="don-cat"  label="Filter by Category" options={cats}  value={donCatFilter}  onChange={setDonCatFilter} />
          <Badge variant="violet">{filtered.length} items shown</Badge>
        </div>

        {/* Donation Table */}
        <div className="table-card" style={{ marginBottom: 24 }}>
          <table className="data-table">
            <thead><tr>
              <Th>Item ID</Th><Th>Product</Th><Th>Days Left</Th>
              <Th>City</Th><Th>NGO</Th><Th>Contact</Th>
              <Th>Status</Th><Th>Actions</Th>
            </tr></thead>
            <tbody>
              {filtered.slice(0, LIMIT).map(r => (
                <tr key={r.item_id}>
                  <Td><span style={{ fontFamily: 'monospace', fontSize: 12, opacity: 0.55 }}>{r.item_id}</span></Td>
                  <Td><strong>{r.product_name}</strong></Td>
                  <Td>
                    <span style={{
                      background: r.days_to_expiry <= 1 ? C.ember : C.sunburst,
                      color: r.days_to_expiry <= 1 ? '#fff' : '#000',
                      border: `1px solid ${C.carbon}`, borderRadius: 1600, padding: '2px 10px',
                      fontSize: 12, fontWeight: 700,
                    }}>{r.days_to_expiry}d</span>
                  </Td>
                  <Td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <MapPin size={12} style={{ opacity: 0.5 }} />
                      {r.city}
                    </div>
                  </Td>
                  <Td><strong>{r.nearest_ngo}</strong></Td>
                  <Td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Phone size={12} style={{ opacity: 0.5 }} />
                      <span style={{ fontSize: 12 }}>{r.ngo_contact}</span>
                    </div>
                  </Td>
                  <Td><DonationBadge status={r.donation_status} /></Td>
                  <Td>
                    {r.donation_status === 'Pending' && (
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button className="btn-cta" style={{ padding: '5px 12px', fontSize: 11 }}
                          onClick={() => markOne(r.item_id, 'Donated')}>
                          ✅ Donate
                        </button>
                        <button className="btn-ghost" style={{ padding: '5px 12px', fontSize: 11 }}
                          onClick={() => markOne(r.item_id, 'Rejected')}>
                          ❌ Reject
                        </button>
                      </div>
                    )}
                    {r.donation_status !== 'Pending' && (
                      <span style={{ fontSize: 12, opacity: 0.4, fontWeight: 500 }}>—</span>
                    )}
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Bulk Actions (app.py ba1–ba3) */}
        {pending.length > 0 && (
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
            <span className="label" style={{ opacity: 0.5 }}>Bulk actions for {pending.length} pending:</span>
            <button className="btn-cta" onClick={() => markAllPending('Donated')}>
              ✅ Mark All Pending → Donated
            </button>
            <button className="btn-ghost" onClick={() => markAllPending('Rejected')}>
              ❌ Mark All Pending → Rejected
            </button>
            <button className="btn-ghost" onClick={exportCSV}>
              <Download size={13} style={{ display: 'inline', marginRight: 6 }} />
              Export CSV
            </button>
          </div>
        )}

        {/* ── MAP — mirrors app.py scatter_mapbox (lines 396-407) ── */}
        {(() => {
          const mapPoints = filtered.filter(r =>
            r.store_latitude && r.store_longitude &&
            !isNaN(r.store_latitude) && !isNaN(r.store_longitude)
          );
          if (!mapPoints.length) return null;
          const statusColor = { Pending: C.sunburst, Donated: C.mint, Rejected: C.ember };
          return (
            <div style={{ marginTop: 32 }}>
              <h3 className="heading-sm" style={{ marginBottom: 16 }}>
                <MapPin size={20} style={{ display: 'inline', marginRight: 8 }} />
                Donation Locations
              </h3>
              <div style={{ borderRadius: 20, border: `1px solid ${C.carbon}`, overflow: 'hidden', height: 450 }}>
                <MapContainer
                  center={[20.5937, 78.9629]}
                  zoom={5}
                  style={{ width: '100%', height: '100%' }}
                  scrollWheelZoom={false}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  {mapPoints.map((r, i) => (
                    <CircleMarker
                      key={`${r.item_id}-${i}`}
                      center={[r.store_latitude, r.store_longitude]}
                      radius={10}
                      pathOptions={{
                        fillColor: statusColor[r.donation_status] || C.mist,
                        fillOpacity: 0.9,
                        color: C.carbon,
                        weight: 1.5,
                      }}
                    >
                      <Popup>
                        <div style={{ fontFamily: 'Inter, sans-serif', fontSize: 13, minWidth: 190 }}>
                          <strong style={{ fontSize: 15 }}>{r.city}</strong><br />
                          <span style={{ opacity: 0.6 }}>{r.product_name}</span><br /><br />
                          <span style={{
                            display: 'inline-block', background: statusColor[r.donation_status] || C.mist,
                            border: '1px solid #000', borderRadius: 999,
                            padding: '1px 9px', fontSize: 11, fontWeight: 700,
                            textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8,
                          }}>{r.donation_status}</span><br />
                          <strong>NGO:</strong> {r.nearest_ngo}<br />
                          <strong>Contact:</strong> {r.ngo_contact}<br />
                          <strong>Category:</strong> {r.category}
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>
              </div>
              <div style={{ display: 'flex', gap: 16, marginTop: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                {[['Pending', C.sunburst], ['Donated', C.mint], ['Rejected', C.ember]].map(([label, color]) => (
                  <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 14, height: 14, borderRadius: '50%', background: color, border: `1px solid ${C.carbon}` }} />
                    <span style={{ fontSize: 12, fontWeight: 700 }}>{label}</span>
                  </div>
                ))}
                <span style={{ fontSize: 12, opacity: 0.45, marginLeft: 4 }}>· Click a marker for details</span>
              </div>
            </div>
          );
        })()}
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════
   FOOTER (app.py lines 412-420)
═══════════════════════════════════════════════════════════════ */
function Footer({ itemCount, dataSource }) {
  return (
    <footer style={{ background: C.carbon, color: C.white, padding: '32px 48px', borderTop: `1px solid rgba(255,255,255,0.08)` }}>
      <div style={{ maxWidth: 1440, margin: '0 auto' }}>
        {/* Dataset Summary (app.py expander) */}
        <div style={{ marginBottom: 20, padding: '16px 20px', background: 'rgba(255,255,255,0.06)', borderRadius: 16, border: '1px solid rgba(255,255,255,0.12)' }}>
          <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
            <div>
              <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.032em', textTransform: 'uppercase', opacity: 0.45 }}>Shape</span>
              <div style={{ fontSize: 14, fontWeight: 700, marginTop: 2 }}>{itemCount.toLocaleString()} rows × 19 columns</div>
            </div>
            <div>
              <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.032em', textTransform: 'uppercase', opacity: 0.45 }}>Source</span>
              <div style={{ fontSize: 14, fontWeight: 700, marginTop: 2 }}>{dataSource}</div>
            </div>
            <div>
              <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.032em', textTransform: 'uppercase', opacity: 0.45 }}>ML Models</span>
              <div style={{ fontSize: 14, fontWeight: 700, marginTop: 2 }}>RandomForest · IsolationForest · Demand Forecasting</div>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: 1600, border: '1px solid rgba(255,255,255,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-lateral)', fontSize: 14, fontWeight: 800 }}>S</div>
            <span style={{ fontWeight: 700 }}>Smart Inventory Dashboard</span>
          </div>
          <span style={{ fontSize: 12, opacity: 0.4 }}>✅ Smart Inventory Dashboard — Sparkathon 2025 | Walmart Hackathon</span>
          <span style={{ fontSize: 12, opacity: 0.4 }}>🔄 Run <code style={{ background: 'rgba(255,255,255,0.1)', padding: '1px 6px', borderRadius: 6 }}>python main.py</code> to refresh analysis data</span>
        </div>
      </div>
    </footer>
  );
}

/* ═══════════════════════════════════════════════════════════════
   ROOT APP
═══════════════════════════════════════════════════════════════ */
export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [filters, setFilters] = useState({ store:'All', stockLevel:'All', expiryRisk:'All', action:'All' });
  const [donFilters, setDonFilters] = useState({ city:'All', cat:'All' });

  // Per-tab data state
  const [kpis,      setKpis]      = useState(null);
  const [fin,       setFin]       = useState(null);
  const [charts,    setCharts]    = useState(null);
  const [urgent,    setUrgent]    = useState(null);
  const [discounts, setDiscounts] = useState(null);
  const [restock,   setRestock]   = useState(null);
  const [remove,    setRemove]    = useState(null);
  const [donations, setDonations] = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [itemCount, setItemCount] = useState(0);

  // Fetch all data whenever filters change
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const [k, f, c, u, d, r, rm, dn] = await Promise.all([
          fetchKPIs(filters),
          fetchFinancial(filters),
          fetchCharts(filters),
          fetchUrgent(filters),
          fetchDiscounts(filters),
          fetchRestock(filters),
          fetchRemove(filters),
          fetchDonations(filters, donFilters.city, donFilters.cat),
        ]);
        if (!cancelled) {
          setKpis(k); setFin(f); setCharts(c);
          setUrgent(u); setDiscounts(d); setRestock(r);
          setRemove(rm); setDonations(dn);
          setItemCount(k.total);
        }
      } catch(e) { console.error('API error:', e); }
      finally { if (!cancelled) setLoading(false); }
    }
    load();
    return () => { cancelled = true; };
  }, [filters, donFilters]);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <MarqueeBand />
      <Nav activeTab={activeTab} setActiveTab={setActiveTab} />

      <div style={{ padding: '24px 48px 0', background: activeTab === 'dashboard' ? 'var(--color-sky-wash)' : '#fff' }}>
        <div style={{ maxWidth: 1440, margin: '0 auto' }}>
          <FilterBar filters={filters} setFilters={setFilters} itemCount={itemCount} />
        </div>
      </div>

      <main style={{ flex: 1 }}>
        {activeTab === 'dashboard' && <DashboardView fin={fin}        kpis={kpis}      charts={charts}    loading={loading} />}
        {activeTab === 'urgent'    && <UrgentView    data={urgent}    loading={loading} />}
        {activeTab === 'discounts' && <DiscountsView  data={discounts} loading={loading} />}
        {activeTab === 'restock'   && <RestockView    data={restock}   loading={loading} />}
        {activeTab === 'remove'    && <RemoveView     data={remove}    loading={loading} />}
        {activeTab === 'donations' && <DonationsView  data={donations} loading={loading} donFilters={donFilters} setDonFilters={setDonFilters} />}
      </main>

      <Footer itemCount={itemCount} dataSource={import.meta.env.VITE_API_URL ? 'Render Backend API' : 'Mock Data (CSV schema)'} />
    </div>
  );
}
