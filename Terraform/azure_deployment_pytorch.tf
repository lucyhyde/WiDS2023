provider "azurerm" {

Enter your Azure credentials below
subscription_id = ""
client_id = ""
client_secret = ""
tenant_id = ""
}

Define the variables required for deployment
variable prefix {
default = "pytorchml"
}

variable location {
default = "eastus"
}

Create a new Resource Group where our resources will be deployed
resource "azurerm_resource_group" "rg" {
name = "${var.prefix}-rg"
location = var.location
}

Create a Public IP Address to allow access into the VM
resource "azurerm_public_ip" "public_ip" {
name = "${var.prefix}-pip"
location = var.location
resource_group_name = azurerm_resource_group.rg.name

AllocationMethod = "Static"
}

Create a Network Security Group and configure rules to allow traffic on specific ports
resource "azurerm_network_security_group" "nsg" {
name = "${var.prefix}-nsg"
location = var.location
resource_group_name = azurerm_resource_group.rg.name

security_rule {
name = "allow-ssh"
priority = 1001
direction = "Inbound"
source_port_range = ""
destination_port_range = "22"
protocol = "Tcp"
access = "Allow"
source_address_prefix = ""
destination_address_prefix = "*"
}

tags = {
Environment = "production"
}
}

Create a Virtual Network andSubnet to host our VM
resource "azurerm_virtual_network" "vnet" {
name = "${var.prefix}-vnet"
location = var.location
resource_group_name = azurerm_resource_group.rg.name
address_spaces = ["192.168.0.0/16"]

subnet {
name = "default"
address_prefixes = ["192

