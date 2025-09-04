# Enhanced Gradient Features for Your Model
# Add these to your notebook to enhance gradient usage

## Enhanced Configuration (add to your CONFIG)
"""
# Add these to your existing CONFIG dictionary:
CONFIG.update({
    'gradient_clip_norm': 1.0,  # Clip gradients to prevent explosion
    'gradient_accumulation_steps': 1,  # Accumulate gradients for larger effective batch size
    'weight_decay': 0.01,  # L2 regularization to prevent overfitting
    'warmup_steps': 100,  # Gradual learning rate increase
    'max_grad_norm': 1.0  # Maximum gradient norm for clipping
})
"""

## Gradient Monitoring and Visualization Class
class GradientMonitor:
    """Monitor and visualize gradients during training"""
    
    def __init__(self):
        self.gradient_norms = []
        self.parameter_norms = []
        self.gradient_histories = {}
        
    def log_gradients(self, model):
        """Log gradient norms for all parameters"""
        total_norm = 0
        param_norm = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm += param.norm().item() ** 2
                grad_norm = param.grad.norm().item() ** 2
                total_norm += grad_norm
                
                # Store gradient history for each parameter
                if name not in self.gradient_histories:
                    self.gradient_histories[name] = []
                self.gradient_histories[name].append(grad_norm)
        
        total_norm = total_norm ** 0.5
        param_norm = param_norm ** 0.5
        
        self.gradient_norms.append(total_norm)
        self.parameter_norms.append(param_norm)
        
        return total_norm, param_norm
    
    def plot_gradients(self):
        """Plot gradient and parameter norms over time"""
        plt.figure(figsize=(15, 5))
        
        # Gradient norms
        plt.subplot(1, 3, 1)
        plt.plot(self.gradient_norms)
        plt.title('Gradient Norms Over Time')
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.grid(True)
        
        # Parameter norms
        plt.subplot(1, 3, 2)
        plt.plot(self.parameter_norms)
        plt.title('Parameter Norms Over Time')
        plt.xlabel('Step')
        plt.ylabel('Parameter Norm')
        plt.yscale('log')
        plt.grid(True)
        
        # Gradient distribution
        plt.subplot(1, 3, 3)
        if self.gradient_norms:
            plt.hist(self.gradient_norms, bins=50, alpha=0.7)
            plt.title('Gradient Norm Distribution')
            plt.xlabel('Gradient Norm')
            plt.ylabel('Frequency')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_gradient_stats(self):
        """Get gradient statistics"""
        if not self.gradient_norms:
            return {}
        
        return {
            'mean_grad_norm': np.mean(self.gradient_norms),
            'std_grad_norm': np.std(self.gradient_norms),
            'max_grad_norm': np.max(self.gradient_norms),
            'min_grad_norm': np.min(self.gradient_norms),
            'gradient_explosion_count': sum(1 for norm in self.gradient_norms if norm > 10),
            'gradient_vanishing_count': sum(1 for norm in self.gradient_norms if norm < 1e-6)
        }

## Enhanced Optimizer with Gradient Features
def create_enhanced_optimizer(model, config):
    """Create optimizer with enhanced gradient features"""
    
    # Separate parameters for different learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    # Create optimizer with gradient features
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    return optimizer

## Enhanced Learning Rate Scheduler
def create_enhanced_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler with warmup"""
    
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    return scheduler

## Gradient Clipping Function
def clip_gradients(model, max_norm):
    """Clip gradients to prevent explosion"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

## Enhanced Training Loop with Gradient Features
def train_with_enhanced_gradients(model, train_loader, val_loader, config):
    """Training loop with enhanced gradient monitoring"""
    
    # Initialize components
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Enhanced optimizer and scheduler
    optimizer = create_enhanced_optimizer(model, config)
    num_training_steps = len(train_loader) * config['num_epochs']
    scheduler = create_enhanced_scheduler(optimizer, config, num_training_steps)
    
    # Gradient monitor
    grad_monitor = GradientMonitor()
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = criterion(outputs.logits.squeeze(), batch['labels'])
            
            # Backward pass
            loss.backward()
            
            # Log gradients before clipping
            grad_norm, param_norm = grad_monitor.log_gradients(model)
            
            # Gradient clipping
            if config['gradient_clip_norm'] > 0:
                clip_gradients(model, config['gradient_clip_norm'])
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Print gradient info every 100 steps
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Gradient Norm: {grad_norm:.4f}")
                print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = criterion(outputs.logits.squeeze(), batch['labels'])
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Show gradient statistics
        grad_stats = grad_monitor.get_gradient_stats()
        print(f"  Gradient Stats: {grad_stats}")
    
    # Plot gradient information
    grad_monitor.plot_gradients()
    
    return model, grad_monitor

## Usage Example
"""
# Add this to your notebook after model initialization:

# Initialize gradient monitor
grad_monitor = GradientMonitor()

# Create enhanced optimizer
optimizer = create_enhanced_optimizer(model, CONFIG)

# Create enhanced scheduler
num_training_steps = len(train_loader) * CONFIG['num_epochs']
scheduler = create_enhanced_scheduler(optimizer, CONFIG, num_training_steps)

# During training, add gradient monitoring:
for epoch in range(CONFIG['num_epochs']):
    for batch in train_loader:
        # ... your existing training code ...
        
        # After backward pass, before optimizer step:
        grad_norm, param_norm = grad_monitor.log_gradients(model)
        
        # Gradient clipping
        if CONFIG['gradient_clip_norm'] > 0:
            clip_gradients(model, CONFIG['gradient_clip_norm'])
        
        # ... rest of training loop ...

# After training, visualize gradients:
grad_monitor.plot_gradients()
grad_stats = grad_monitor.get_gradient_stats()
print("Gradient Statistics:", grad_stats)
""" 