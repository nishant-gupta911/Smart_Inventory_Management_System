#!/usr/bin/env python3
"""
Utility functions for retail inventory management system with donation functionality.

This module provides utility functions for handling donation-related operations,
data processing, and analysis in the inventory management system.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def update_donation_status(df: pd.DataFrame, item_id: str, new_status: str) -> pd.DataFrame:
    """
    Update the donation status for a specific item.
    
    Args:
        df (pd.DataFrame): The inventory DataFrame
        item_id (str): The ID of the item to update
        new_status (str): New donation status ("Pending", "Donated", "Rejected")
    
    Returns:
        pd.DataFrame: Modified DataFrame with updated donation status
    
    Raises:
        ValueError: If item_id not found or item not donation eligible
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_updated = df.copy()
        
        # Find the item
        item_mask = df_updated['item_id'] == item_id
        
        if not item_mask.any():
            raise ValueError(f"Item with ID '{item_id}' not found in DataFrame")
        
        # Check if item is donation eligible
        if not df_updated.loc[item_mask, 'donation_eligible'].iloc[0]:
            raise ValueError(f"Item '{item_id}' is not donation eligible")
        
        # Validate new status
        valid_statuses = ["Pending", "Donated", "Rejected", ""]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid donation status '{new_status}'. Must be one of: {valid_statuses}")
        
        # Update the donation status
        df_updated.loc[item_mask, 'donation_status'] = new_status
        
        logger.info(f"Updated donation status for item '{item_id}' to '{new_status}'")
        return df_updated
        
    except Exception as e:
        logger.error(f"Error updating donation status: {str(e)}")
        raise


def get_nearest_ngo(city_name: str) -> Dict[str, str]:
    """
    Get the nearest NGO details for a given city.
    
    Args:
        city_name (str): Name of the city
    
    Returns:
        Dict[str, str]: Dictionary with NGO name, address, and contact details
    """
    try:
        # Hardcoded dictionary of Indian cities and nearby NGOs
        city_ngo_mapping = {
            'Mumbai': {
                'name': 'Mumbai Food Bank',
                'address': '123 SV Road, Bandra West, Mumbai, Maharashtra 400050',
                'contact': 'contact@mumbaifoodbank.org'
            },
            'Delhi': {
                'name': 'Delhi Food Banking Network',
                'address': '456 CP, Connaught Place, New Delhi, Delhi 110001',
                'contact': 'info@delhifoodbank.org'
            },
            'Bangalore': {
                'name': 'Bangalore Food Rescue',
                'address': '789 MG Road, Bangalore, Karnataka 560001',
                'contact': 'help@bangalorefoodrescue.org'
            },
            'Chennai': {
                'name': 'Chennai Hunger Relief',
                'address': '321 Anna Salai, Chennai, Tamil Nadu 600002',
                'contact': 'support@chennaihungerrelief.org'
            },
            'Hyderabad': {
                'name': 'Hyderabad Food Foundation',
                'address': '654 Banjara Hills, Hyderabad, Telangana 500034',
                'contact': 'contact@hyderabadfood.org'
            },
            'Pune': {
                'name': 'Pune Meal Mission',
                'address': '987 FC Road, Pune, Maharashtra 411005',
                'contact': 'info@punemeal.org'
            },
            'Kolkata': {
                'name': 'Kolkata Food Bridge',
                'address': '147 Park Street, Kolkata, West Bengal 700016',
                'contact': 'help@kolkatafoodbridge.org'
            },
            'Ahmedabad': {
                'name': 'Ahmedabad Food Connect',
                'address': '258 CG Road, Ahmedabad, Gujarat 380009',
                'contact': 'connect@ahmedabadfood.org'
            },
            'Jaipur': {
                'name': 'Jaipur Food Care',
                'address': '369 MI Road, Jaipur, Rajasthan 302001',
                'contact': 'care@jaipurfood.org'
            },
            'Surat': {
                'name': 'Surat Food Share',
                'address': '741 Ring Road, Surat, Gujarat 395002',
                'contact': 'share@suratfood.org'
            }
        }
        
        # Default NGOs for cities not in the mapping
        default_ngos = [
            {
                'name': 'Food Bank Central',
                'address': 'Central Location, India',
                'contact': 'contact@foodbankcentral.org'
            },
            {
                'name': 'Helping Hands Foundation',
                'address': 'Community Center, India',
                'contact': 'help@helpinghands.org'
            },
            {
                'name': 'Care Alliance Network',
                'address': 'Service Hub, India',
                'contact': 'care@carealliance.org'
            },
            {
                'name': 'Hope Foundation',
                'address': 'Unity Plaza, India',
                'contact': 'hope@hopefoundation.org'
            },
            {
                'name': 'Community Kitchen',
                'address': 'Local Center, India',
                'contact': 'kitchen@communitykitchen.org'
            }
        ]
        
        # Clean city name for matching
        city_clean = city_name.strip().title()
        
        # Return mapped NGO or random default
        if city_clean in city_ngo_mapping:
            ngo_info = city_ngo_mapping[city_clean]
        else:
            # Use random choice for unmapped cities
            import random
            ngo_info = random.choice(default_ngos)
            logger.info(f"City '{city_name}' not found in mapping, using default NGO: {ngo_info['name']}")
        
        return {
            'nearest_ngo': ngo_info['name'],
            'ngo_address': ngo_info['address'],
            'ngo_contact': ngo_info['contact']
        }
        
    except Exception as e:
        logger.error(f"Error getting nearest NGO for city '{city_name}': {str(e)}")
        # Return default NGO on error
        return {
            'nearest_ngo': 'Food Bank Central',
            'ngo_address': 'Central Location, India',
            'ngo_contact': 'contact@foodbankcentral.org'
        }


def get_donation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive donation summary statistics.
    
    Args:
        df (pd.DataFrame): The inventory DataFrame
    
    Returns:
        Dict[str, Any]: Dictionary containing donation summary statistics
    """
    try:
        # Ensure required columns exist
        if 'donation_eligible' not in df.columns:
            logger.warning("donation_eligible column not found, assuming no items are eligible")
            df = df.copy()
            df['donation_eligible'] = False
        
        if 'donation_status' not in df.columns:
            logger.warning("donation_status column not found, assuming empty status")
            df = df.copy()
            df['donation_status'] = ""
        
        # Fill NaN values
        df = df.copy()
        df['donation_eligible'] = df['donation_eligible'].fillna(False)
        df['donation_status'] = df['donation_status'].fillna("")
        
        # Filter donation-eligible items
        donation_eligible_df = df[df['donation_eligible'] == True]
        
        # Basic summary statistics
        total_donation_eligible = len(donation_eligible_df)
        
        # Count by donation status
        status_counts = donation_eligible_df['donation_status'].value_counts().to_dict()
        
        # Count by category and city
        category_city_breakdown = []
        if 'category' in df.columns and 'city' in df.columns:
            category_city_counts = donation_eligible_df.groupby(['category', 'city']).size().reset_index(name='count')
            category_city_breakdown = category_city_counts.to_dict('records')
        elif 'category' in df.columns:
            category_counts = donation_eligible_df['category'].value_counts().to_dict()
            category_city_breakdown = [{'category': k, 'count': v} for k, v in category_counts.items()]
        
        # Top NGOs receiving donations
        top_ngos = {}
        if 'nearest_ngo' in df.columns:
            donated_items = donation_eligible_df[donation_eligible_df['donation_status'] == 'Donated']
            if len(donated_items) > 0:
                ngo_counts = donated_items['nearest_ngo'].value_counts().head(10)
                top_ngos = ngo_counts.to_dict()
        
        # Create summary dictionary
        summary = {
            'total_donation_eligible': total_donation_eligible,
            'total_items': len(df),
            'donation_eligible_percentage': round((total_donation_eligible / len(df)) * 100, 2) if len(df) > 0 else 0,
            'donation_status_counts': status_counts,
            'category_city_breakdown': category_city_breakdown,
            'top_ngos': top_ngos,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Generated donation summary: {total_donation_eligible} eligible items out of {len(df)} total")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating donation summary: {str(e)}")
        return {
            'total_donation_eligible': 0,
            'total_items': len(df) if df is not None else 0,
            'donation_eligible_percentage': 0,
            'donation_status_counts': {},
            'category_city_breakdown': [],
            'top_ngos': {},
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }


def filter_pending_donations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter items with pending donation status.
    
    Args:
        df (pd.DataFrame): The inventory DataFrame
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only items with pending donations
    """
    try:
        # Ensure required columns exist
        if 'donation_eligible' not in df.columns:
            logger.warning("donation_eligible column not found, assuming no items are eligible")
            df = df.copy()
            df['donation_eligible'] = False
        
        if 'donation_status' not in df.columns:
            logger.warning("donation_status column not found, assuming empty status")
            df = df.copy()
            df['donation_status'] = ""
        
        # Fill NaN values
        df_clean = df.copy()
        df_clean['donation_eligible'] = df_clean['donation_eligible'].fillna(False)
        df_clean['donation_status'] = df_clean['donation_status'].fillna("")
        
        # Filter for pending donations
        pending_donations = df_clean[
            (df_clean['donation_eligible'] == True) & 
            (df_clean['donation_status'] == 'Pending')
        ].copy()
        
        logger.info(f"Found {len(pending_donations)} items with pending donation status")
        return pending_donations
        
    except Exception as e:
        logger.error(f"Error filtering pending donations: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error


def save_updated_inventory(df: pd.DataFrame, output_path: str) -> bool:
    """
    Save the updated inventory DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The inventory DataFrame to save
        output_path (str): Path where the CSV file should be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save DataFrame to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved inventory data to {output_path}")
        print(f"✅ Inventory data saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving inventory data to {output_path}: {str(e)}")
        print(f"❌ Failed to save inventory data: {str(e)}")
        return False


def validate_donation_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that all required donation columns exist in the DataFrame.
    
    Args:
        df (pd.DataFrame): The inventory DataFrame to validate
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_missing_columns)
    """
    required_columns = [
        'donation_eligible',
        'donation_status',
        'store_latitude',
        'store_longitude',
        'nearest_ngo',
        'ngo_address',
        'ngo_contact'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    
    if not is_valid:
        logger.warning(f"Missing donation columns: {missing_columns}")
    
    return is_valid, missing_columns


def add_missing_donation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing donation columns with default values.
    
    Args:
        df (pd.DataFrame): The inventory DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with all donation columns added
    """
    try:
        df_updated = df.copy()
        
        # Add donation_eligible if missing
        if 'donation_eligible' not in df_updated.columns:
            df_updated['donation_eligible'] = False
            logger.info("Added donation_eligible column with default value False")
        
        # Add donation_status if missing
        if 'donation_status' not in df_updated.columns:
            df_updated['donation_status'] = ""
            logger.info("Added donation_status column with default empty string")
        
        # Add store coordinates if missing
        if 'store_latitude' not in df_updated.columns:
            df_updated['store_latitude'] = np.random.uniform(8.0, 37.0, len(df_updated))  # India latitude range
            logger.info("Added store_latitude column with random values")
        
        if 'store_longitude' not in df_updated.columns:
            df_updated['store_longitude'] = np.random.uniform(68.0, 97.0, len(df_updated))  # India longitude range
            logger.info("Added store_longitude column with random values")
        
        # Add NGO details if missing
        if 'nearest_ngo' not in df_updated.columns:
            cities = df_updated.get('city', pd.Series(['Delhi'] * len(df_updated)))
            ngo_details = cities.apply(lambda city: get_nearest_ngo(city))
            
            df_updated['nearest_ngo'] = ngo_details.apply(lambda x: x['nearest_ngo'])
            df_updated['ngo_address'] = ngo_details.apply(lambda x: x['ngo_address'])
            df_updated['ngo_contact'] = ngo_details.apply(lambda x: x['ngo_contact'])
            
            logger.info("Added NGO detail columns based on city mapping")
        
        return df_updated
        
    except Exception as e:
        logger.error(f"Error adding missing donation columns: {str(e)}")
        return df


def calculate_donation_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate key donation performance metrics.
    
    Args:
        df (pd.DataFrame): The inventory DataFrame
    
    Returns:
        Dict[str, float]: Dictionary containing donation metrics
    """
    try:
        # Ensure required columns exist
        df_clean = add_missing_donation_columns(df)
        
        total_items = len(df_clean)
        donation_eligible = (df_clean['donation_eligible'] == True).sum()
        donated_items = ((df_clean['donation_eligible'] == True) & 
                        (df_clean['donation_status'] == 'Donated')).sum()
        pending_items = ((df_clean['donation_eligible'] == True) & 
                        (df_clean['donation_status'] == 'Pending')).sum()
        rejected_items = ((df_clean['donation_eligible'] == True) & 
                         (df_clean['donation_status'] == 'Rejected')).sum()
        
        metrics = {
            'donation_rate': (donation_eligible / total_items * 100) if total_items > 0 else 0,
            'success_rate': (donated_items / donation_eligible * 100) if donation_eligible > 0 else 0,
            'pending_rate': (pending_items / donation_eligible * 100) if donation_eligible > 0 else 0,
            'rejection_rate': (rejected_items / donation_eligible * 100) if donation_eligible > 0 else 0,
            'total_items': float(total_items),
            'donation_eligible_items': float(donation_eligible),
            'donated_items': float(donated_items),
            'pending_items': float(pending_items),
            'rejected_items': float(rejected_items)
        }
        
        logger.info(f"Calculated donation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating donation metrics: {str(e)}")
        return {
            'donation_rate': 0.0,
            'success_rate': 0.0,
            'pending_rate': 0.0,
            'rejection_rate': 0.0,
            'total_items': 0.0,
            'donation_eligible_items': 0.0,
            'donated_items': 0.0,
            'pending_items': 0.0,
            'rejected_items': 0.0
        }


# Additional utility functions for common operations

def format_donation_report(summary: Dict[str, Any]) -> str:
    """
    Format donation summary into a readable report string.
    
    Args:
        summary (Dict[str, Any]): Donation summary from get_donation_summary()
    
    Returns:
        str: Formatted report string
    """
    try:
        report_lines = [
            "📊 DONATION SUMMARY REPORT",
            "=" * 50,
            f"📦 Total Items: {summary.get('total_items', 0):,}",
            f"🤝 Donation Eligible: {summary.get('total_donation_eligible', 0):,} ({summary.get('donation_eligible_percentage', 0)}%)",
            "",
            "📋 Status Breakdown:",
        ]
        
        status_counts = summary.get('donation_status_counts', {})
        for status, count in status_counts.items():
            if status:  # Skip empty status
                report_lines.append(f"   {status}: {count:,} items")
        
        top_ngos = summary.get('top_ngos', {})
        if top_ngos:
            report_lines.extend([
                "",
                "🏆 Top NGOs:",
            ])
            for ngo, count in list(top_ngos.items())[:5]:
                report_lines.append(f"   {ngo}: {count} donations")
        
        report_lines.extend([
            "",
            f"📅 Generated: {summary.get('generated_at', 'Unknown')}",
            "=" * 50
        ])
        
        return "\n".join(report_lines)
        
    except Exception as e:
        logger.error(f"Error formatting donation report: {str(e)}")
        return f"Error generating report: {str(e)}"


# Export the main functions for easy importing
__all__ = [
    'update_donation_status',
    'get_nearest_ngo', 
    'get_donation_summary',
    'filter_pending_donations',
    'save_updated_inventory',
    'validate_donation_columns',
    'add_missing_donation_columns',
    'calculate_donation_metrics',
    'format_donation_report'
]
